"""Per-request reasoning-effort steering of raon-1-mini MoE outputs.

The checkpoint carries the steering vectors as ordinary tensors "model.layers.<L>.block_sparse_moe.reasoning_effort_steering"
(HF release naming: "model.layers.<L>.moe.reasoning_effort_steering"), F32 [hidden_size] at alpha 1, and the alpha per
effort and the steered layers in config.json "reasoning_effort_steering": {"alpha":
{...}, "layers": [...], "default": <effort>}. While vLLM verifies the model config,
configure() sets RAON_REASONING_EFFORT_STEERING to {"alpha": {effort: alpha}, "layers":
[L, ...], "default": <effort or null>, "id": <checkpoint identity>}, which the engine
processes inherit. Unset, every function here is a no-op and the model is unchanged
(load_weights then drops steering tensors). A request with reasoning effort e adds
alpha[e] * c_L to the MoE output of each listed layer. A request without an effort uses
the default effort (alpha 0 when none is declared). "none" turns thinking off in the
chat template and is never steered. The effort names are those of the alpha table, e.g.
low, medium, high, xhigh and max; older checkpoints that declare low, medium and high keep
working.

  API server  chat completions and /v1/responses: the effort the chat template
              sees selects alpha, which travels as extra_args[ALPHA_KEY] tagged with a per-process secret.
              Unknown efforts are rejected with HTTP 400, and so are steered efforts with beam search or on
              /v1/chat/completions/batch, whose requests do not carry the tagged alpha.
  engine      InputProcessor.process_inputs (every endpoint) keeps alpha only when the tag is present, so a
              client cannot set it through vllm_xargs, and salts the prefix cache per alpha value: requests
              share cached KV/KDA state only when they were steered identically. alpha 0 keeps the request's
              own salt; client salts in the reserved SALT_PREFIX namespace are rejected (HTTP 400, or an error
              event on a stream). The tag stays in the API server's request log when request logging is on.
  runners     G, one persistent fp32 GPU buffer [max_num_batched_tokens] whose address CUDA graphs capture,
              holds each scheduled token's alpha (0 on padding). It is filled after prepare_inputs.
  model       the decoder layer passes each steered layer's MoE output through steer():
              out = bf16(fp32(moe_out) + G[t] * c_L); G[t] = 0 returns moe_out's bits (including -0). It equals
              folding the add into the shared + routed combine (whose result is the MoE output), and it does not depend on the MoE backend (fp32 combine, stock fused MoE, BF16, MXFP8 or NVFP4 experts).

TP1/DP1 without prefill context parallelism only; attach() refuses anything else. The DFlash draft has no Raon MoE layer and is not steered; it
reads the steered target hidden states.
"""

import json
import math
import os
import re

import numpy as np
import torch

ENV = "RAON_REASONING_EFFORT_STEERING"
ALPHA_KEY = "raon_reasoning_effort_alpha"
SALT_PREFIX = "raon-effort-"
_MARK = "raon_reasoning_effort_tag"
_SECRET = os.urandom(16).hex()  # per API process; only api_request_error() can produce it


def _config():
    raw = os.environ.get(ENV)
    if not raw:
        return None
    config = json.loads(raw)
    alpha = config["alpha"]
    assert set(config) == {"alpha", "layers", "default", "id"}, config
    assert isinstance(alpha, dict), config
    assert config["default"] is None or config["default"] in alpha, config
    assert config["layers"] and all(type(L) is int and L >= 0 for L in config["layers"]), config
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in alpha.values()), alpha
    return config


CONFIG = _config()
G: torch.Tensor | None = None  # per-token alpha, shared by every steered layer
_vectors: dict[int, torch.Tensor] = {}  # layer -> on-device F32 [hidden] buffer, filled by load_vector()
_loaded: set[int] = set()
KEY_RE = re.compile(r"model\.layers\.(\d+)\.(?:block_sparse_moe|moe)\.reasoning_effort_steering")


NO_THINKING = "none"  # thinking off in the chat template; never steered


def alpha_for(effort) -> float:
    """reasoning effort -> alpha; ValueError for an effort the deployment does not define."""
    if effort is None:
        effort = CONFIG["default"]
    if effort is None or effort == NO_THINKING:
        return 0.0
    if effort not in CONFIG["alpha"]:
        names = ", ".join(dict.fromkeys([NO_THINKING, *CONFIG["alpha"]]))
        raise ValueError(f"reasoning_effort must be one of {names}; got {effort!r}")
    return float(CONFIG["alpha"][effort])


def cache_salt(alpha: float, salt: str | None) -> str | None:
    """Prefix-cache salt of a request steered with alpha (unchanged for alpha 0)."""
    if (salt or "").startswith(SALT_PREFIX):
        raise ValueError(f"cache_salt prefix {SALT_PREFIX!r} is reserved for reasoning-effort steering")
    if not alpha:
        return salt
    prefix = f"{SALT_PREFIX}{CONFIG['id']}-{alpha!r}"
    return prefix + (f"|{salt}" if salt else "")


def _request_effort_alpha(serving, request) -> float:
    effort = serving._effective_chat_template_kwargs(request).get("reasoning_effort")
    return alpha_for(None if effort is None else str(effort))


def _tag_request(serving, request):
    """API entry: the effort the template sees -> tagged alpha; a client-sent alpha is dropped."""
    alpha = _request_effort_alpha(serving, request)
    if alpha and getattr(request, "use_beam_search", False):
        raise ValueError("reasoning_effort steering is not supported with beam search")
    xargs = {k: v for k, v in (request.vllm_xargs or {}).items() if k not in (ALPHA_KEY, _MARK)}
    if alpha:
        xargs.update({ALPHA_KEY: alpha, _MARK: _SECRET})
    request.vllm_xargs = xargs or None


def _engine_request(request):
    """InputProcessor, every endpoint: keep alpha only when tagged, and salt the request by it."""
    sp, alpha = request.sampling_params, 0.0
    if sp is not None and sp.extra_args:
        extra = dict(sp.extra_args)
        if extra.pop(_MARK, None) != _SECRET:
            extra.pop(ALPHA_KEY, None)
        sp.extra_args = extra or None
        alpha = float(extra.get(ALPHA_KEY, 0.0))
    request.cache_salt = cache_salt(alpha, request.cache_salt)
    return request


def request_alpha(sp) -> float:
    return float((sp.extra_args or {}).get(ALPHA_KEY, 0.0)) if sp is not None else 0.0


def configure(hf_config, model: str) -> None:
    """API server, while vLLM verifies the model config: enable steering when config.json declares it.

    Sets RAON_REASONING_EFFORT_STEERING, which the engine and worker processes inherit, from
    config.json "reasoning_effort_steering": {"alpha": {effort: alpha, ...},
    "layers": [...], "default": effort}. "default" is optional and must name an
    alpha entry. The checkpoint's weights carry the vectors; load_weights checks
    that every listed layer received one.
    """
    global CONFIG
    steering = getattr(hf_config, "reasoning_effort_steering", None)
    if not steering:
        return
    alpha, layers = steering.get("alpha"), steering.get("layers")
    default = steering.get("default")
    if (not isinstance(alpha, dict) or not alpha
            or any(not isinstance(k, str) for k in alpha)
            or alpha.get(NO_THINKING, 0) != 0
            or any(type(v) not in (int, float) or not 0 <= v < float("inf") for v in alpha.values())
            or not layers or not all(type(L) is int and L >= 0 for L in layers)
            or (default is not None and (default not in alpha or default == NO_THINKING))):
        raise ValueError(
            "config.json reasoning_effort_steering needs a non-negative alpha per "
            f"effort name ({NO_THINKING!r} only with 0), the steered layers, and a "
            f"default that names an alpha entry other than {NO_THINKING!r}"
        )
    identity = json.dumps({"model": model, "steering": steering}, sort_keys=True)
    value = json.dumps({"alpha": alpha, "layers": sorted(layers), "default": default,
                        "id": __import__("hashlib").sha256(identity.encode()).hexdigest()[:16]}, sort_keys=True)
    os.environ[ENV] = value
    CONFIG = _config()


# Hooks called by vLLM itself (entrypoints, InputProcessor, GPU model runners). Each is a no-op without steering.

def api_request_error(serving, request) -> str | None:
    """Chat completions and /v1/responses: tag the request's alpha; an error message for a 400, or None."""
    if CONFIG is None:
        return None
    try:
        _tag_request(serving, request)
    except ValueError as error:
        return str(error)
    return None


def batch_request_error(serving, request) -> str | None:
    """/v1/chat/completions/batch: its conversations do not carry the tagged alpha, so steered efforts are refused."""
    if CONFIG is None:
        return None
    try:
        alpha = _request_effort_alpha(serving, request.to_chat_completion_request(request.messages[0]))
    except ValueError as error:
        return str(error)
    return "reasoning_effort steering is not supported on /v1/chat/completions/batch" if alpha else None


def engine_request(request):
    """InputProcessor.process_inputs result: keep only tagged alpha and salt the prefix cache by it."""
    if CONFIG is None:
        return request
    from vllm.exceptions import VLLMValidationError
    try:
        return _engine_request(request)
    except ValueError as error:  # AsyncLLM turns a plain ValueError into a 500
        raise VLLMValidationError(str(error), parameter="cache_salt") from error


def fill(runner, alpha_req: np.ndarray, qsl: torch.Tensor, n_pad: int) -> None:
    """alpha_req: fp32 alpha per batch request (batch order); qsl: GPU query_start_loc[:num_reqs + 1]."""
    n = len(alpha_req)
    if not alpha_req.any():
        if getattr(runner, "_raon_effort_dirty", False):
            G.zero_()
            runner._raon_effort_dirty = False
        return
    alpha = torch.from_numpy(np.append(alpha_req, 0).astype(np.float32)).to(G.device, non_blocking=True)
    tokens = torch.arange(n_pad, device=G.device, dtype=qsl.dtype)
    req = torch.searchsorted(qsl[1: n + 1].contiguous(), tokens, right=True)  # token -> request; n = padding
    G.zero_()
    G[:n_pad] = alpha[req]
    runner._raon_effort_dirty = True


def runner_v2_add_requests(runner, scheduler_output) -> None:
    """Model Runner V2 add_requests: remember each new request's alpha by its slot."""
    if G is None:
        return
    if not hasattr(runner, "_raon_effort_alpha"):
        runner._raon_effort_alpha = np.zeros(runner.max_num_reqs, dtype=np.float32)
    for r in scheduler_output.scheduled_new_reqs:
        runner._raon_effort_alpha[runner.req_states.req_id_to_index[r.req_id]] = request_alpha(r.sampling_params)


def runner_v2_prepare_inputs(runner, batch) -> None:
    """Model Runner V2 prepare_inputs: give every scheduled token its request's alpha."""
    if G is None:
        return
    fill(runner, runner._raon_effort_alpha[batch.idx_mapping_np[: batch.num_reqs]], batch.query_start_loc,
         batch.num_tokens_after_padding)


def runner_v1_prepare_inputs(runner) -> None:
    """V1 GPUModelRunner _prepare_inputs: same, padding to the whole buffer (V1 pads to an unknown size)."""
    if G is None:
        return
    n = runner.input_batch.num_reqs
    alpha = np.fromiter((request_alpha(runner.requests[r].sampling_params)
                         for r in runner.input_batch.req_ids[:n]), np.float32, n)
    fill(runner, alpha, runner.query_start_loc.gpu[: n + 1], G.numel())


def attach(moe, layer_idx: int, hidden_size: int, vllm_config) -> None:
    """Model init: give layer_idx's MoE block a zero vector buffer that load_vector() fills (moe=None: dense layer)."""
    global G
    if CONFIG is None or layer_idx not in CONFIG["layers"]:
        return
    if G is None:
        parallel = vllm_config.parallel_config
        # The MoE output is per token on a single rank only; PCP splits tokens after the gate is filled.
        assert (parallel.tensor_parallel_size == 1 and parallel.data_parallel_size == 1
                and parallel.prefill_context_parallel_size == 1), (
            f"{ENV} supports TP1/DP1 without prefill context parallelism")
    assert moe is not None, f"{ENV} lists dense layer {layer_idx}"
    device = torch.device("cuda", torch.cuda.current_device())
    if G is None:
        G = torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens, dtype=torch.float32, device=device)
    moe.raon_effort_c = _vectors[layer_idx] = torch.zeros(hidden_size, dtype=torch.float32, device=device)


def check_attached() -> None:
    """End of model init: every listed layer must be a MoE layer of this model."""
    if CONFIG is not None:
        missing = sorted(set(CONFIG["layers"]) - set(_vectors))
        assert not missing, f"{ENV}: layers {missing} are not MoE layers of this model"


def take_vectors(weights):
    """load_weights filter: route steering tensors to load_vector() and yield every other weight unchanged."""
    for name, tensor in weights:
        m = KEY_RE.fullmatch(name)
        if m is None:
            yield name, tensor
        elif CONFIG is not None:  # steering off: the vectors are simply not used
            load_vector(int(m.group(1)), tensor)


def load_vector(layer_idx: int, tensor: torch.Tensor) -> None:
    assert layer_idx in _vectors, f"{ENV}: checkpoint has a steering vector for layer {layer_idx}, not in {CONFIG['layers']}"
    c = _vectors[layer_idx]
    assert tensor.dtype == torch.float32 and tuple(tensor.shape) == tuple(c.shape), (
        f"{ENV}: layer {layer_idx} vector must be F32 {list(c.shape)}, got {tensor.dtype} {list(tensor.shape)}")
    assert torch.isfinite(tensor).all(), f"{ENV}: non-finite vector for layer {layer_idx}"  # NaN * 0 is NaN
    c.copy_(tensor)
    _loaded.add(layer_idx)
    print(f"[effort-steering] layer {layer_idx}: MoE output += alpha*c |c|={tensor.norm():.4f} "
          f"alpha={CONFIG['alpha']} gate={G.numel()}", flush=True)


def check_loaded() -> None:
    """After loading weights: every listed layer received its vector (an unloaded zero vector would not steer)."""
    if CONFIG is not None:
        missing = sorted(set(CONFIG["layers"]) - _loaded)
        assert not missing, f"{ENV}: checkpoint has no steering vector for layers {missing}"


def steer(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """A steered layer's MoE output [T, H] with each token's alpha * c added in fp32 and rounded once.

    Plain torch so torch.compile fuses it into one elementwise kernel inside the decoder layer's graph (a custom
    Triton kernel would split the graph). G[t] = 0 selects h itself, so unsteered tokens keep their bits.
    """
    T = h.shape[0]
    g = G[:T, None]
    return torch.where(g == 0, h, (h.float() + g * c).to(h.dtype))
