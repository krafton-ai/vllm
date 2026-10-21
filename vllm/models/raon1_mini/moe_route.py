# SPDX-License-Identifier: Apache-2.0
"""RAON_MOE_ROUTE_CAPTURE: record the eager MoE forward's top-k selection.

Records, per MoE layer, the ``topk_ids`` tensor that
``_raon_fp32_combine_moe_forward`` dispatched on, and writes it next to the
parity dump so a training-side forward can replay the same selection through
Megatron-Core's ``RouterReplay``
(``megatron/core/transformer/moe/router_replay.py``).

Why this capture site and not vLLM's ``enable_return_routed_experts``
--------------------------------------------------------------------
Only this site records the routing the dumped activations were computed from:

  * ``enable_return_routed_experts`` binds a ``RoutedExpertsCapturer`` onto the
    FusedMoE routers and surfaces the result through the request/output API
    (``RoutedExpertsReader.get_routed_experts``). The parity path replaces
    ``KimiMoE.forward`` with the eager ``_raon_fp32_combine_moe_forward``, so
    the fused path the capturer is designed around is not the path that
    produced ``L*.moe``.
  * The layer probe drives the model with ``nn.Module`` hooks and never reads
    the request-level output plumbing.
  * Recording ``topk_ids`` where the expert dispatch consumes it makes the
    replayed indices the same object the dumped activation was built from.

Default off: ``model.py`` guards the call with a module-level constant
resolved once at import, so the disabled path runs no tensor op.

Output layout (side dir ``routing``)
------------------------------------
``${PARITY_DUMP_ROOT}/routing/L{ii:02d}.route.safetensors`` — key ``"t"``,
shape ``[probe_length, topk]``, int64, global 0-based layer index ``ii`` (the
same index the ``L{ii:02d}.moe`` site uses), plus an atomically written
``meta.json``. Comparison tools read only the ``trainer`` and ``engine`` side
dirs, so the routing artifact is not compared as an activation site.

The writer runs at interpreter exit (the probe is a single-process run,
``VLLM_ENABLE_V1_MULTIPROCESSING=0``). An atexit failure cannot change the
process exit status, so callers should verify the routing dir against the
probe file's ``moe_layers`` metadata.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

_ENV = "RAON_MOE_ROUTE_CAPTURE"
_WEIGHTS_ENV = "RAON_MOE_ROUTE_CAPTURE_WEIGHTS"
_ROOT_ENV = "PARITY_DUMP_ROOT"
_LENGTH_ENV = "RAON_PROBE_LENGTH"

SIDE = "routing"
SUFFIX = ".route"
PROBS_SUFFIX = ".rprobs"


def enabled() -> bool:
    return os.environ.get(_ENV, "0") not in ("0", "", "false")


def weights_enabled() -> bool:
    """RAON_MOE_ROUTE_CAPTURE_WEIGHTS: also record the top-k weights.

    Default off. The capture is a detach/copy and changes no engine
    arithmetic either way.

    Replaying only the ids still leaves the routing weights to each side's
    router gate GEMM. Different GEMM realizations of the gate (e.g. TE
    ``te_general_gemm`` vs ``torch.mm(x, weight.T, out_dtype=fp32)``) differ
    by tens of fp32 ulp, enough to flip a weight across a bf16 rounding
    boundary. Carrying the weights in the replay payload pins them the same
    way the ids are pinned.
    """
    return os.environ.get(_WEIGHTS_ENV, "0") not in ("0", "", "false")


def site_name(layer_idx: int) -> str:
    return f"L{layer_idx:02d}{SUFFIX}"


def probs_site_name(layer_idx: int) -> str:
    return f"L{layer_idx:02d}{PROBS_SUFFIX}"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{_ENV}=1 requires {name} to be set: the capture has nowhere to "
            "write and no probe length to trim to. Refusing to run a "
            "capture whose routing artifact would be silently absent."
        )
    return value


_ENABLED = enabled()
_WEIGHTS = weights_enabled()
# Resolved once, at import, so a misconfiguration fails before the model is
# built rather than at interpreter exit.
_ROOT = Path(_require_env(_ROOT_ENV)) if _ENABLED else None
_PROBE_LENGTH = int(_require_env(_LENGTH_ENV)) if _ENABLED else 0

if _WEIGHTS and not _ENABLED:
    raise RuntimeError(
        f"{_WEIGHTS_ENV}=1 without {_ENV}=1: the weight payload rides on the "
        "routing capture and has nowhere to be written. Refusing to run a "
        "capture whose weight artifact would be silently absent.")

# layer index -> token count of EVERY forward that layer saw (provenance: a
# vLLM run also does a memory-profiling forward and may do warmups, and the
# metadata records which forward was kept).
_forwards: dict[int, list[int]] = {}
# layer index -> the last forward wide enough to hold the probe row.
_selected: dict[int, torch.Tensor] = {}
_selected_probs: dict[int, torch.Tensor] = {}


def record(
    layer_idx: int,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
) -> None:
    """Record one MoE layer's top-k selection (and, when
    ``RAON_MOE_ROUTE_CAPTURE_WEIGHTS=1``, the weights attached to it) for one
    forward."""
    if not _ENABLED:
        return
    num_tokens = int(topk_ids.shape[0])
    _forwards.setdefault(int(layer_idx), []).append(num_tokens)
    if num_tokens >= _PROBE_LENGTH:
        # Only the probe row's prefill (and the profiling forward) are this
        # wide; the last such forward is the probe row's. Kept on CPU so the
        # tensor does not pin device memory for the rest of the run.
        _selected[int(layer_idx)] = (
            topk_ids[:_PROBE_LENGTH].detach().to(torch.int64).cpu().contiguous()
        )
        if _WEIGHTS:
            if topk_weights is None:
                raise RuntimeError(
                    f"{_WEIGHTS_ENV}=1 but the dispatch site passed no "
                    "topk_weights -- the capture would write an ids-only "
                    "payload that would be replayed as if it carried "
                    "weights")
            if tuple(topk_weights.shape) != tuple(topk_ids.shape):
                raise RuntimeError(
                    f"topk_weights {tuple(topk_weights.shape)} does not match "
                    f"topk_ids {tuple(topk_ids.shape)}: the weights would not "
                    "be aligned to the selection they belong to")
            # fp32 on the wire: these are the fp32 weights the expert chain
            # folds to bf16, saved without an extra rounding.
            _selected_probs[int(layer_idx)] = (
                topk_weights[:_PROBE_LENGTH]
                .detach().to(torch.float32).cpu().contiguous()
            )


def write(root: Path | None = None) -> list[str]:
    """Write every captured layer's selection. Returns the site names."""
    if not _ENABLED:
        return []
    out_dir = Path(root or _ROOT) / SIDE
    out_dir.mkdir(parents=True, exist_ok=True)

    sites: list[str] = []
    probs_sites: list[str] = []
    topk: set[int] = set()
    for layer_idx in sorted(_selected):
        tensor = _selected[layer_idx]
        if tuple(tensor.shape)[0] != _PROBE_LENGTH:
            raise RuntimeError(
                f"layer {layer_idx}: captured {tensor.shape[0]} token rows, "
                f"probe length is {_PROBE_LENGTH}"
            )
        topk.add(int(tensor.shape[1]))
        site = site_name(layer_idx)
        save_file({"t": tensor}, str(out_dir / f"{site}.safetensors"))
        sites.append(site)
        if _WEIGHTS:
            probs = _selected_probs.get(layer_idx)
            if probs is None or tuple(probs.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"layer {layer_idx}: RAON_MOE_ROUTE_CAPTURE_WEIGHTS=1 but "
                    f"the weight payload is "
                    f"{None if probs is None else tuple(probs.shape)} against "
                    f"ids {tuple(tensor.shape)}")
            psite = probs_site_name(layer_idx)
            save_file({"t": probs}, str(out_dir / f"{psite}.safetensors"))
            # Kept out of `sites`: meta["sites"] lists only the .route files,
            # which consumers cross-check against the files on disk. The
            # weight payload gets its own key.
            probs_sites.append(psite)

    if len(topk) > 1:
        raise RuntimeError(f"MoE layers disagree about topk: {sorted(topk)}")

    meta = {
        "model_sha": os.environ.get("PARITY_MODEL_SHA", ""),
        "dtype": "int64",
        "sites": sites,
        "probe_length": _PROBE_LENGTH,
        "topk": (sorted(topk)[0] if topk else 0),
        "lever": _ENV,
        # RAON_MOE_ROUTE_CAPTURE_WEIGHTS payload, listed separately so
        # meta["sites"] stays the .route census the consumers cross-check.
        "weights_lever": _WEIGHTS_ENV,
        "weights_captured": bool(_WEIGHTS),
        "probs_sites": probs_sites,
        "probs_dtype": "float32",
        "source": "vllm.models.raon1_mini.moe_route (eager _raon_fp32_combine_moe_forward)",
        # Every forward each layer saw, so readers can tell which one was
        # kept.
        "forward_token_counts": {str(k): v for k, v in sorted(_forwards.items())},
    }
    tmp = out_dir / f".meta.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, out_dir / "meta.json")  # atomic file rename
    print(
        f"[moe_route] wrote {len(sites)} routing sites"
        f"{f' + {len(probs_sites)} weight sites' if _WEIGHTS else ''} "
        f"-> {out_dir} (probe_length={_PROBE_LENGTH}, topk={meta['topk']})",
        flush=True,
    )
    return sites


def _write_at_exit() -> None:
    if not _ENABLED:
        return
    try:
        write()
    except Exception as exc:  # atexit cannot fail the process; callers re-check
        print(f"[moe_route] CAPTURE WRITE FAILED: {exc!r}", flush=True)


if _ENABLED:
    atexit.register(_write_at_exit)
    print(
        f"[moe_route] {_ENV}=1 — capturing eager MoE topk_ids"
        f"{' AND topk_weights (' + _WEIGHTS_ENV + '=1)' if _WEIGHTS else ''} "
        f"(probe_length={_PROBE_LENGTH}, out={_ROOT}/{SIDE})",
        flush=True,
    )
