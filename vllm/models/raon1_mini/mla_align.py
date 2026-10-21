"""MLA prefill alignment with the trainer's attention realization.

Trainer and engine MLA prefill realizations:

  * trainer: Megatron ``TorchSDPADotProductAttention(kernel="flash_attn4")``
    -> ``flash_attn.cute.flash_attn_func`` (FA4 CuTe DSL), ``num_splits=1``,
    causal, batch layout.
  * engine: vLLM 0.28 ``FlashAttnPrefillBackend`` -> ``flash_attn_varlen_func``
    with ``fa_version=2`` when FA4 is unavailable: on Blackwell the
    auto-selected version is FA4, but vLLM's vendored FA4 CuTe path imports
    ``quack``, and ``quack_kernels 0.6.4`` fails to import against
    ``nvidia_cutlass_dsl 4.6.0.dev0`` (``cutlass.pipeline`` has no
    ``alloc_reserved_mbarrier``). With ``RAON_FA_PREFILL_VERSION=2`` the
    engine runs a different kernel family from the trainer on every MLA
    prefill.

This module closes that gap engine-side, in two independently gated steps:

1. ``RAON_QUACK_COMPAT_PATH`` (default unset): a directory whose ``quack``
   package is importable against the installed cutlass DSL (e.g.
   ``quack_kernels 0.5.3``). If the environment's own ``import quack`` fails
   and this path is set, the compatible ``quack`` is pre-imported from it
   into ``sys.modules`` so vLLM's vendored FA4 (``vllm_flash_attn.cute``)
   becomes importable. The path is inserted only for that one import and
   removed immediately; nothing else resolves from it.

2. ``RAON_MLA_PREFILL_FA4_NS1`` (default "0", opt-in): forces the MLA
   prefill backend's flash-attn calls to ``num_splits=1``, the trainer's own
   setting and the batch-shape-invariant reduction. Without
   VLLM_BATCH_INVARIANT vLLM leaves ``num_splits=0`` (auto) on this path
   (mla/prefill/flash_attn.py).

With both active and ``RAON_FA_PREFILL_VERSION`` unset (so Blackwell
auto-selects FA4), the engine MLA prefill core realization is
FA4-CuTe/num_splits=1/varlen, which is bitwise identical to the trainer's
FA4 call on the same q/k/v at the RAON shape.

Both knobs are feature-detected and fail loud-but-open: a missing symbol on a
vLLM bump logs a warning naming the lost guarantee and the engine still
starts.
"""

import logging
import os
import sys

from . import raon_phase

logger = logging.getLogger(__name__)


def preload_compat_quack() -> None:
    """Step 1: make ``import quack`` work if the installed copy cannot.

    No-op when ``quack`` already imports or when ``RAON_QUACK_COMPAT_PATH``
    is unset. Never shadows a working installed quack.
    """
    compat = os.environ.get("RAON_QUACK_COMPAT_PATH")
    if "quack" in sys.modules:
        return
    try:
        import quack  # noqa: F401  -- installed copy works; nothing to do
        return
    except Exception as venv_exc:
        if not compat:
            logger.warning(
                "raon_vllm: `import quack` fails (%s) and "
                "RAON_QUACK_COMPAT_PATH is unset -- vLLM's FA4 CuTe paths "
                "(MLA prefill FA4, QuACK norms) stay unimportable",
                venv_exc,
            )
            return
    sys.modules.pop("quack", None)
    sys.path.insert(0, compat)
    try:
        import quack  # noqa: F401
        # The CuTe DSL AST preprocessor re-executes kernel-source imports at
        # compile time via attribute access on the already-imported package
        # (cutlass base_dsl ast_preprocessor.exec_imports) -- `from quack
        # import copy_utils` there fails with "module 'quack' has no attribute
        # 'copy_utils'" unless the submodule objects are already bound. Bind
        # the set vllm_flash_attn/cute uses.
        import importlib

        for sub in ("copy_utils", "layout_utils", "sm90_utils",
                    "compile_utils", "cute_dsl_utils", "activation"):
            try:
                mod = importlib.import_module(f"quack.{sub}")
                # quack's DSL import machinery (quack.dsl / cute_dsl_ptxas
                # loader) can leave the parent attribute unbound even though
                # the submodule sits in sys.modules; the cutlass AST
                # preprocessor resolves via getattr, so bind explicitly.
                if not hasattr(quack, sub):
                    setattr(quack, sub, mod)
            except Exception as sub_exc:
                logger.warning(
                    "raon_vllm: quack.%s preload failed (%s) -- a CuTe kernel "
                    "needing it will fail at compile", sub, sub_exc,
                )
        logger.info(
            "raon_vllm: quack pre-imported from RAON_QUACK_COMPAT_PATH=%s "
            "(installed copy fails to import against the cutlass DSL)",
            compat,
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        logger.warning(
            "raon_vllm: RAON_QUACK_COMPAT_PATH=%s quack import FAILED (%s); "
            "FA4 CuTe paths stay unimportable", compat, exc,
        )
    finally:
        sys.path.remove(compat)


def install_mla_prefill_ns1_pin() -> None:
    """Step 2: pin the MLA prefill flash-attn calls to ``num_splits=1``.

    Wraps ``FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims`` so both
    ``run_prefill_new_tokens`` and ``run_prefill_context_chunk`` inherit the
    pin -- the same single-seam property vLLM itself relies on for the
    VLLM_BATCH_INVARIANT case of this function (mla/prefill/flash_attn.py);
    this pin is that branch, scoped to RAON without the global env.
    """
    if os.environ.get("RAON_MLA_PREFILL_FA4_NS1", "0") != "1":
        return
    try:
        from vllm.v1.attention.backends.mla.prefill import flash_attn as _fa_mod
    except Exception as exc:  # pragma: no cover - import-time guard
        logger.warning(
            "raon_vllm: MLA prefill num_splits=1 pin SKIPPED -- could not "
            "import vllm...mla.prefill.flash_attn (%s); trainer-paired MLA "
            "prefill is NOT guaranteed on this build", exc,
        )
        return
    backend = getattr(_fa_mod, "FlashAttnPrefillBackend", None)
    inner = getattr(backend, "_flash_attn_varlen_diff_headdims", None)
    if backend is None or inner is None:
        logger.warning(
            "raon_vllm: MLA prefill num_splits=1 pin SKIPPED -- "
            "FlashAttnPrefillBackend._flash_attn_varlen_diff_headdims not "
            "found (vLLM version drift); validate the prefill seam against "
            "this vLLM before claiming MLA prefill parity",
        )
        return
    if getattr(backend, "_raon_ns1_pin", False):
        return

    def _pinned(self, *args, **kwargs):
        kwargs.setdefault("num_splits", 1)
        return inner(self, *args, **kwargs)

    backend._flash_attn_varlen_diff_headdims = _pinned
    backend._raon_ns1_pin = True
    logger.info(
        "raon_vllm: MLA prefill pinned to num_splits=1 "
        "(RAON_MLA_PREFILL_FA4_NS1=1; trainer-paired, batch-invariant)",
    )


_GATE_ENV = "RAON_MLA_OUTPUT_GATE_TWO_ROUND"


class InertLeverError(RuntimeError):
    """Raised when a flag provably cannot change a bit."""

# Hand-computed bf16 tie case for the output-gate rounding boundary.
#   x = 0.423828125, g = -2.25  (both exact bf16)
#   sigmoid_fp32(g)                       = 0.09534946084022522
#   round to bf16, then multiply, round    -> 0.040283203125
#   keep fp32, multiply, round once        -> 0.040527343750
# One bf16 ulp apart. This is the bit the flag exists to move, so the check
# fails the import when the two forms agree -- which is what an
# implementation that drops the intermediate rounding would do.
_GATE_TIE_X = 0.423828125
_GATE_TIE_G = -2.25
_GATE_TIE_TWO_ROUND = 0.040283203125
_GATE_TIE_ONE_ROUND = 0.04052734375


def _is_real_forward(attn_out) -> bool:
    """True for a forward whose attention output can distinguish the two gates.

    vLLM's memory-profiling and warmup passes reach this seam with
    ``attn_metadata is None``, where ``_attention`` returns after
    ``attn_out.zero_()`` -- and ``0 * sigmoid(g)`` is 0 under both roundings,
    so the announce below would report an inertness that is a property of the
    profile pass, not of the flag. Zero and non-finite outputs are therefore
    skipped and the announce lands on the first forward that can actually
    tell the two realizations apart.

    Graph-capture safety: the ``.all()``/``.any()`` reads below sync the
    host, which is illegal inside CUDA-graph stream capture, and capture at
    engine init precedes every real forward (e.g. under
    VLLM_USE_BREAKABLE_CUDAGRAPH this raises
    cudaErrorStreamCaptureUnsupported). A captured forward is never the
    announce's target (the announce is a one-time host-side diagnostic), so
    a capturing stream reports "not a real forward" and the announce defers
    to the first non-captured real forward. The gate's replacement
    arithmetic itself (``attn_out * gate.sigmoid()``) is pure device work and
    captures fine.
    """
    import torch

    if torch.cuda.is_current_stream_capturing():
        return False
    return bool(torch.isfinite(attn_out).all()) and bool((attn_out != 0).any())


def _assert_gate_rounding_boundary_is_real() -> None:
    """Import-time, CPU-only, fail-closed: prove the flag can change a bit.

    A flag that cannot change a bit is a silent no-op, so the arithmetic
    claim is checked here rather than trusted. Three assertions:

      1. the two roundings really disagree on the tie case (the boundary
         exists at all);
      2. eager ``x * g.sigmoid()`` -- what the replacement below computes --
         lands on the two-rounding value, i.e. on Megatron's
         ``fused_output_gate`` semantics;
      3. it does not land on the one-rounding value, i.e. it is not what
         inductor's fusion of the same expression produces.
    """
    import torch

    x = torch.tensor([_GATE_TIE_X], dtype=torch.bfloat16)
    g = torch.tensor([_GATE_TIE_G], dtype=torch.bfloat16)
    s32 = torch.sigmoid(g.float())
    two = (x.float() * s32.bfloat16().float()).bfloat16()
    one = (x.float() * s32).bfloat16()
    if two.item() != _GATE_TIE_TWO_ROUND or one.item() != _GATE_TIE_ONE_ROUND:
        raise RuntimeError(
            f"{_GATE_ENV}: output-gate tie case moved -- expected two-round "
            f"{_GATE_TIE_TWO_ROUND!r} / one-round {_GATE_TIE_ONE_ROUND!r}, "
            f"measured {two.item()!r} / {one.item()!r}. Re-derive the "
            "constants before trusting this flag")
    if two.item() == one.item():
        raise RuntimeError(
            f"{_GATE_ENV}: the two gate roundings agree on the tie case, so "
            "this flag cannot change a bit -- refusing to install a silent "
            "no-op")
    eager = x * g.sigmoid()
    if eager.item() != two.item():
        raise RuntimeError(
            f"{_GATE_ENV}: eager `x * g.sigmoid()` is {eager.item()!r}, not "
            f"the trainer's two-rounding {two.item()!r}; the replacement "
            "below would install the wrong realization")
    if eager.item() == one.item():
        raise RuntimeError(
            f"{_GATE_ENV}: eager `x * g.sigmoid()` equals the single-rounding "
            "value, so replacing the compiled gate with it is a no-op")


def install_mla_output_gate_two_round() -> None:
    """Step 3: restore the trainer's two roundings in the MLA output gate.

    vLLM computes the gate as

        @torch.compile(backend=current_platform.simple_compile_backend)
        def _gate_sigmoid_mul(attn_out, gate):
            return attn_out * gate.sigmoid()

    (``vllm/models/kimi_k3/nvidia/mla.py``). Under ``torch.compile`` inductor
    fuses the sigmoid and the multiply into one pointwise kernel that keeps the
    sigmoid in fp32 and rounds once, at the store::

        bf16( fp32(attn_out) * sigmoid_fp32(gate) )

    The trainer does not. Megatron runs the gate through its own Triton kernel
    (``megatron/core/fusions/fused_output_gate.py``, enabled by
    ``--mla-fused-output-gate``) which rounds the sigmoid to the activation
    dtype before the product, on purpose -- its docstring says "Rounding
    follows the unfused aten sequence: sigmoid is computed in fp32 and
    rounded to the activation dtype before the product, matching the
    released Kimi MLA semantics"::

        bf16( fp32(attn_out) * fp32( bf16(sigmoid_fp32(gate)) ) )

    That is one extra bf16 rounding per element, which can differ by up to
    one bf16 ulp of the output on a large fraction of elements.

    The flag rebinds the module global to an ordinary, uncompiled eager
    implementation. Eager torch already has the trainer's semantics
    (``gate.sigmoid()`` on a bf16 tensor rounds to bf16 before the multiply);
    what the flag removes is inductor's fusion of the two ops. Default off
    means the module global is untouched and the engine is byte-identical.
    """
    # RAON_PREFILL_ALIGNED: install whenever the phase gate is enabled (even
    # if the raw env is off), so the per-call selector below has something
    # to switch to during prefill; the raw env alone still controls decode.
    if os.environ.get(_GATE_ENV, "0") != "1" and not raon_phase.PREFILL_ALIGNED:
        return
    _assert_gate_rounding_boundary_is_real()
    try:
        from . import mla_attention as _mla_mod
    except Exception as exc:  # pragma: no cover - import-time guard
        logger.warning(
            "raon_vllm: %s SKIPPED -- could not import "
            "raon_vllm.mla_attention (%s); the MLA output gate keeps "
            "vLLM's single-rounding fused realization", _GATE_ENV, exc,
        )
        return
    compiled = getattr(_mla_mod, "_gate_sigmoid_mul", None)
    if compiled is None:
        logger.warning(
            "raon_vllm: %s SKIPPED -- _gate_sigmoid_mul not found (vLLM "
            "version drift); validate the MLA output gate against this vLLM "
            "before claiming MLA parity", _GATE_ENV,
        )
        return
    if getattr(_mla_mod, "_raon_gate_two_round", False):
        return

    # Fail closed on a silently inert flag. The only thing this flag does is
    # take the gate out of inductor's fusion, so if the shipped symbol is not
    # a torch.compile wrapper there is nothing to undo; say so loudly rather
    # than install quietly.
    if not any(hasattr(compiled, attr) for attr in
               ("_torchdynamo_orig_callable", "_torchdynamo_inline",
                "get_compiler_config", "dynamo_ctx")):
        logger.warning(
            "raon_vllm: %s -- vLLM's _gate_sigmoid_mul is NOT a torch.compile "
            "wrapper on this build (%r), so eager already runs it and this "
            "flag cannot change a bit. Installing anyway so the runtime "
            "announce below records the measured delta, but treat a zero "
            "there as an inert flag, not as parity",
            _GATE_ENV, type(compiled).__name__,
        )

    state = {"announced": False}

    def _gate_wanted() -> bool:
        # RAON_PREFILL_ALIGNED: `compiled` (the original vLLM fused gate,
        # bound above at install time) is preserved in this closure. The raw
        # env or the per-forward phase verdict decides which realization this
        # call takes; with the raw env off, decode keeps returning to
        # `compiled` below even while the phase gate is enabled.
        return (os.environ.get(_GATE_ENV, "0") == "1"
                or raon_phase.aligned("mla.output_gate"))

    def _two_round_gate(attn_out, gate):
        if not _gate_wanted():
            return compiled(attn_out, gate)
        # `gate.sigmoid()` on a bf16 tensor materializes a bf16 sigmoid -- the
        # rounding Megatron's kernel performs and inductor's fusion elides.
        gated = attn_out * gate.sigmoid()
        if not state["announced"] and _is_real_forward(attn_out):
            state["announced"] = True
            try:
                fused = compiled(attn_out, gate)
                delta = (gated.float() - fused.float()).abs()
                n_diff = int((delta > 0).sum())
                # Fail closed on the load-bearing assumption: inductor fuses
                # this bf16 pointwise chain in fp32 and rounds once (torch's
                # `codegen_upcast_to_fp32`). That is a property of the
                # installed torch build, so it is asserted against the real
                # compiled function on the real tensors. A synthetic
                # import-time `torch.compile` of the same expression would
                # test a kernel this engine does not run.
                if n_diff == 0:
                    # With the flag set explicitly, a zero delta is an error.
                    # Under the phase-aligned composition, dummy-load warmup
                    # can have no difference (and inductor may be disabled).
                    # The chosen path still materializes the bf16 sigmoid
                    # before multiplication, matching the trainer, so a zero
                    # observed delta does not abort warmup.
                    if (not raon_phase.PREFILL_ALIGNED
                            or os.environ.get(_GATE_ENV, "0") == "1"):
                        raise InertLeverError(
                            f"{_GATE_ENV}=1 is INERT: vLLM's torch.compile'd "
                            "output gate already agrees bitwise with the "
                            "two-rounding form on real tensors, so this flag "
                            "has no effect. Either inductor is not fusing "
                            "the sigmoid on this torch build or the flag "
                            "failed to displace the compiled path")
                    logger.warning(
                        "[parity] aligned MLA output gate: first measurable "
                        "warmup has zero observed delta versus vLLM; "
                        "trainer two-rounding is still executed, but this "
                        "warmup does not establish a nonzero effect"
                    )
                else:
                    logger.warning(
                        "[parity] %s=1: MLA output gate rounds sigmoid to "
                        "bf16 before the product -- vs vLLM's compiled gate "
                        "max|d|=%.6e mean|d|=%.6e over %d/%d elements",
                        _GATE_ENV, delta.max().item(), delta.mean().item(),
                        n_diff, delta.numel(),
                    )
                    print(
                        f"[parity] {_GATE_ENV}=1: MLA output gate rounds "
                        f"sigmoid to bf16 before the product -- vs vLLM's "
                        f"compiled gate max|d|={delta.max().item():.6e} "
                        f"mean|d|={delta.mean().item():.6e} over "
                        f"{n_diff}/{delta.numel()} elements", flush=True,
                    )
            except InertLeverError:
                raise
            except Exception as exc:  # pragma: no cover - announce only
                logger.warning(
                    "raon_vllm: %s announce failed (%s); the flag is live "
                    "but unmeasured", _GATE_ENV, exc,
                )
        return gated

    _mla_mod._gate_sigmoid_mul = _two_round_gate
    _mla_mod._raon_gate_two_round = True
    logger.info(
        "raon_vllm: MLA output gate pinned to the trainer's two-rounding "
        "realization (%s=1; megatron fused_output_gate semantics)", _GATE_ENV,
    )


_LATENT_NORM_ENV = "RAON_MLA_LATENT_NORM_QUACK"


def install_mla_latent_norm_quack() -> None:
    """``RAON_MLA_LATENT_NORM_QUACK=1`` (default off): run the MLA latent q/kv
    RMSNorms on the trainer's QuACK ``rmsnorm`` kernel instead of vLLM's fused
    Triton ``fused_q_kv_rmsnorm``.

    The two latent-norm kernels use the same fp32 formula but a different
    row-reduction tree and rsqrt, so they can disagree by one bf16 ulp on
    individual elements; the q up-projection and attention then spread that
    difference across the token. The trainer's q/kv layernorm is QuACK, so
    the engine adopts the identical kernel. RAON_VLLM_QUACK_RMSNORM does not
    affect this call site: it wraps the RMSNorm module forward, while the
    MLA block calls the function directly on ``.weight.data``. Patching the
    module global covers every call site (prefill and decode share the
    realization).
    """
    # RAON_PREFILL_ALIGNED: install whenever the phase gate is enabled, as
    # for the output gate above, so the per-call selector below has
    # something to switch to.
    if os.environ.get(_LATENT_NORM_ENV, "0") != "1" and not raon_phase.PREFILL_ALIGNED:
        return
    # Fail closed: a run that asks for the flag must get it.
    from quack import rmsnorm as _quack_rmsnorm
    from . import mla_attention as _mla_mod

    if getattr(_mla_mod, "_raon_latent_norm_quack", False):
        return
    _orig_fused_q_kv_rmsnorm = getattr(_mla_mod, "fused_q_kv_rmsnorm", None)
    if not callable(_orig_fused_q_kv_rmsnorm):
        raise RuntimeError(
            f"{_LATENT_NORM_ENV}=1: raon_vllm.mla_attention.fused_q_kv_rmsnorm not "
            "found (vLLM version drift); refusing to run unpaired"
        )

    def _latent_norm_wanted() -> bool:
        # RAON_PREFILL_ALIGNED: the original (vLLM's fused_q_kv_rmsnorm) is
        # captured into `_orig_fused_q_kv_rmsnorm` above. The raw env or the
        # per-forward phase verdict decides which realization this call takes.
        return (os.environ.get(_LATENT_NORM_ENV, "0") == "1"
                or raon_phase.aligned("mla.latent_norm"))

    def _quack_q_kv_rmsnorm(q_c, kv_c, q_w, kv_w, eps):
        if not _latent_norm_wanted():
            return _orig_fused_q_kv_rmsnorm(q_c, kv_c, q_w, kv_w, eps)
        # Matches the trainer: one QuACK call per section, contiguous rows,
        # same eps.
        return (
            _quack_rmsnorm(q_c.contiguous(), q_w, eps=eps),
            _quack_rmsnorm(kv_c.contiguous(), kv_w, eps=eps),
        )

    _mla_mod.fused_q_kv_rmsnorm = _quack_q_kv_rmsnorm
    _mla_mod._raon_latent_norm_quack = True
    print(
        f"[raon-mla] {_LATENT_NORM_ENV}=1: MLA latent q/kv RMSNorm -> "
        "quack.rmsnorm (the trainer's QuACK "
        "RMSNorm kernel)",
        flush=True,
    )


def install() -> None:
    preload_compat_quack()
    install_mla_prefill_ns1_pin()
    install_mla_output_gate_two_round()
    install_mla_latent_norm_quack()
