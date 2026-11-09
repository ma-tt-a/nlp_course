import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_configs import Qwen35Config, make_small_config


# ============================================================================
# RMSNorm (note the non-standard, zero-init-centered variant used here)
# ============================================================================

class RMSNorm(nn.Module):
    """Qwen-style RMSNorm: zero-init weight, output = normalized * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * (1 + self.weight)
        return x.to(dtype)


class RMSNormGated(nn.Module):
    """RMSNorm (weight init to ones) followed by an elementwise `* silu(gate)`."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate):
        dtype = x.dtype
        x = x.float()
        x = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x.to(dtype)
        x = self.weight * x * F.silu(gate.float())
        x = x.to(dtype)
        return x


# ============================================================================
# Rotary position embedding (partial RoPE)
# ============================================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # split x in half along the last dim: x1, x2
    # return cat(-x2, x1) along the last dim
    x1, x2 = torch.chunk(x, chunks=2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Qwen35RotaryEmbedding(nn.Module):
    """Precomputes rotary cos/sin caches from position_ids for partial RoPE.

    `__init__` is given below. See homework.ipynb for the `.forward` formula.
    `hidden_states` is only used for its dtype/device -- cast the returned
    cos/sin to it, or bf16/fp16 models will silently upcast q/k to float32.
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        dtype, device = hidden_states.dtype, hidden_states.device
        freqs = torch.einsum("ij, k -> ijk", position_ids, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = torch.cos(emb).to(dtype).to(device)
        sin = torch.sin(emb).to(dtype).to(device)
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies partial RoPE to q and k; see homework.ipynb for the derivation."""
    # only the first rotary_dim channels of q/k get rotated; the rest pass through unchanged
    # rotated = x * cos + rotate_half(x) * sin, applied to that leading slice, then re-concatenated
    # cos/sin need a head dim inserted to broadcast against q/k's (b, heads, s, dim) shape
    rotate = lambda x_rot: x_rot * cos[:, None, ...] + rotate_half(x_rot) * sin[:, None, ...]
    rotary_dim = cos.shape[-1]
    q_rot = torch.cat([rotate(q[..., :rotary_dim]), q[..., rotary_dim:]], dim=-1)
    k_rot = torch.cat([rotate(k[..., :rotary_dim]), k[..., rotary_dim:]], dim=-1)
    return q_rot, k_rot


# ============================================================================
# Full attention: regular MHA + QK-norm + partial RoPE + output gate
# GQA is an optional extension, discussed in week07_efficiency.
# ============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Given helper for the GQA bonus; regular attention does not need it.

    (batch, num_kv_heads, seq, dim) -> (batch, num_kv_heads * n_rep, seq, dim).
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, dim)


def build_causal_mask(seq_len: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Additive float causal mask of shape (1, 1, seq_len, seq_len)."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


class GatedAttention(nn.Module):
    """Regular multi-head attention with QK-norm, partial RoPE and output gate.

    The required task uses equal numbers of query, key and value heads.
    Fewer key/value heads use the optional forward_gqa bonus. The lecture in
    week07_efficiency will explain GQA and many more efficiency techniques.

    `__init__` is given below. See tests/test_qwen_public.py::TestGatedAttention
    for exact attributes/shapes referenced by `.forward`.
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _project_qkv_and_gate(self, hidden_states, position_embeddings):
        """Given: shared plumbing (projections + per-head QK-norm + RoPE).

        Returns query_states (b, num_attention_heads, s, head_dim), key_states
        and value_states (b, num_key_value_heads, s, head_dim), and gate
        (b, s, num_attention_heads * head_dim). All head counts are equal for
        the required regular-attention task.
        """
        batch_size, seq_len, _ = hidden_states.shape

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim * 2),
            2, dim=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states, gate

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        """TODO: regular multi-head attention, with one Q/K/V head per group.

        Must be vectorized: no Python loop over heads, no
        `F.scaled_dot_product_attention` -- see homework.ipynb.
        Keep the supplied optional-bonus dispatch below. The required path
        has equal Q/K/V head counts and needs no key/value repetition.
        """
        if self.num_key_value_heads != self.num_attention_heads:
            return self.forward_gqa(hidden_states, position_embeddings, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device
        query_states, key_states, value_states, gate = self._project_qkv_and_gate(hidden_states, position_embeddings)
        mask = build_causal_mask(seq_len, dtype, device) if attention_mask is None else attention_mask
        scores = query_states @ key_states.transpose(-2, -1) / self.head_dim ** 0.5 + mask
        attn = F.softmax(scores, dim=-1) @ value_states
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.head_dim * self.num_attention_heads)
        out = self.o_proj(attn * F.sigmoid(gate))
        return out

    def forward_gqa(self, hidden_states, position_embeddings, attention_mask=None):
        """Bonus TODO: grouped-query attention for fewer KV heads than Q heads.

        Query head h uses key/value head h // num_key_value_groups. Honor
        the same mask, QK-norm, RoPE and output gate as regular attention;
        vectorize the heads without SDPA. The supplied repeat_kv helper may
        be useful. This extension is needed for actual Qwen3.5 checkpoints,
        but not for any required forward task or the other two bonuses.

        GQA and many more efficiency techniques will be explained in the
        week07_efficiency lecture.
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device
        query_states, key_states, value_states, gate = self._project_qkv_and_gate(hidden_states, position_embeddings)
        mask = build_causal_mask(seq_len, dtype, device) if attention_mask is None else attention_mask
        num_repeat = self.num_key_value_groups
        k = key_states.repeat_interleave(num_repeat, dim=1)
        v = value_states.repeat_interleave(num_repeat, dim=1)
        scores = query_states @ k.transpose(-2, -1) / self.head_dim ** 0.5 + mask
        attn = F.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.head_dim * self.num_attention_heads)
        out = self.o_proj(attn * F.sigmoid(gate))
        return out

@torch.no_grad()
def naive_looped_gated_attention(attn: GatedAttention, hidden_states, position_embeddings, attention_mask=None):
    """Given, fully-implemented reference: numerically identical to
    `GatedAttention.forward`, but with an explicit Python for-loop over heads.
    """
    batch_size, seq_len, _ = hidden_states.shape
    query_states, key_states, value_states, gate = attn._project_qkv_and_gate(hidden_states, position_embeddings)

    head_outputs = []
    for head_idx in range(attn.num_attention_heads):
        kv_head_idx = head_idx // attn.num_key_value_groups
        q_h = query_states[:, head_idx : head_idx + 1]
        k_h = key_states[:, kv_head_idx : kv_head_idx + 1]
        v_h = value_states[:, kv_head_idx : kv_head_idx + 1]
        # Use the same unfused operations required of students. Comparing a
        # manual implementation to fused SDPA measures backend choice as well
        # as vectorization, and can penalize a correct implementation on CPU.
        scores = (q_h @ k_h.transpose(-2, -1)) * attn.scaling
        mask = attention_mask
        if mask is None:
            mask = build_causal_mask(seq_len, scores.dtype, scores.device)
        elif mask.shape[1] != 1:
            mask = mask[:, head_idx : head_idx + 1]
        weights = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q_h.dtype)
        head_outputs.append(weights @ v_h)

    attn_output = torch.cat(head_outputs, dim=1)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    attn_output = attn_output * torch.sigmoid(gate)
    return attn.o_proj(attn_output)


# ============================================================================
# SwiGLU MLP
# ============================================================================

class MLP(nn.Module):
    def __init__(self, config: Qwen35Config, intermediate_size: int = None):
        super().__init__()
        intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Gated DeltaNet (linear attention)
# ============================================================================

def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last dimension of x. Given utility, not a blank."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gated_delta_rule_recurrent(query, key, value, g, beta):
    """Given, fully implemented: delta-rule recurrence via an O(seq_len) Python loop.

    Shapes: query/key/value (batch, seq_len, num_heads, dim); g, beta (batch, seq_len, num_heads).
    Differentiable: callers may use torch.no_grad() for inference, but the
    recurrence itself must preserve gradients during training.
    """
    initial_dtype = query.dtype
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = (x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g))

    batch_size, num_heads, seq_len, head_k_dim = key.shape
    head_v_dim = value.shape[-1]
    scale = 1.0 / (head_k_dim ** 0.5)
    query = query * scale

    state = torch.zeros(batch_size, num_heads, head_k_dim, head_v_dim, dtype=value.dtype, device=value.device)
    core_attn_out = torch.zeros(batch_size, num_heads, seq_len, head_v_dim, dtype=value.dtype, device=value.device)

    for t in range(seq_len):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        decay_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)

        state = state * decay_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet (linear attention).

    `__init__` is given below. See tests/test_qwen_public.py::TestGatedDeltaNet
    for exact module names/shapes referenced by `.forward`.
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim, padding=self.conv_kernel_size - 1,
        )

        self.in_proj_qkv = nn.Linear(config.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0, 16)))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        """Project q/k/v, convolve causally, and apply the gated delta rule.

        1. in_proj_qkv -> (b, conv_dim, s) -> depthwise conv1d -> keep
           the first s positions -> SiLU -> transpose to (b, s, conv_dim).
        2. Split [key_dim, key_dim, value_dim] and reshape q/k to
           (b, s, num_k_heads, head_k_dim), v to (b, s, num_v_heads, head_v_dim).
           Repeat each q/k head num_v_heads // num_k_heads times.
        3. beta = sigmoid(in_proj_b(x)); compute decay in float32 as
           g = -exp(A_log) * softplus(in_proj_a(x).float() + dt_bias).
        4. Run the provided recurrent function. Reshape in_proj_z(x) to
           (b, s, num_v_heads, head_v_dim) for the gated normalization.
        5. Apply norm to each value head and its gate, flatten the heads,
           then out_proj to (b, s, hidden_size). Preserve autograd throughout.

        See homework.ipynb for the equations and dtype requirements.
        """
        batch_size, seq_len, _ = hidden_states.shape
        proj_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        proj_qkv = F.silu(self.conv1d(proj_qkv)[..., :seq_len]).transpose(1, 2)
        q, k, v = proj_qkv.split(split_size=[self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        num_repeat = self.num_v_heads // self.num_k_heads
        q = q.repeat_interleave(num_repeat, dim=2)
        k = k.repeat_interleave(num_repeat, dim=2)
        beta = F.sigmoid(self.in_proj_b(hidden_states))
        g = -torch.exp(self.A_log.float()) * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
        core_out = gated_delta_rule_recurrent(q, k, v, g, beta)
        z = self.in_proj_z(hidden_states).view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        out = self.out_proj(self.norm(core_out, z).view(batch_size, seq_len, -1))
        return out


# ============================================================================
# Decoder layer, model, causal LM
# ============================================================================

class Qwen35DecoderLayer(nn.Module):
    """Pre-LN residual block; dispatches to full or linear attention.

    `__init__` is given below. See tests/test_qwen_public.py::TestDecoderLayer
    for exact attribute names referenced by `.forward`.
    """

    def __init__(self, config: Qwen35Config, layer_idx: int):
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "full_attention":
            self.self_attn = GatedAttention(config, layer_idx)
        else:
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        """Applies the two pre-LN residual connections; see homework.ipynb for the formula."""
        h = self.input_layernorm(hidden_states)
        if self.block_type == "full_attention":
            h = hidden_states + self.self_attn(h, position_embeddings, attention_mask)
        else:
            h = hidden_states + self.linear_attn(h)
        out = self.post_attention_layernorm(h)
        return h + self.mlp(out)
        


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len) int64
        Returns last_hidden_state: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        embs = self.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        for layer in self.layers:
            pos_embs = self.rotary_emb(embs, position_ids)
            embs = layer(embs, pos_embs)
        return self.norm(embs)
        


class Qwen35ForCausalLM(nn.Module):
    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.config = config
        self.model = Qwen35Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, labels=None):
        """input_ids: (batch, seq_len); labels: (batch, seq_len), optional.

        Returns (logits, loss), with loss=None when labels is absent.
        Labels are already aligned with the predictions; do not shift them
        internally. For next-token training pass tokens[:, :-1] as input_ids
        and tokens[:, 1:] as labels. Cross-entropy ignores labels equal to -100.
        `self.lm_head` is a real nn.Linear, weight-aliased to
        `embed_tokens.weight` when tied -- so this works unchanged for both
        tied and untied checkpoints.
        """
        batch_size, seq_len = input_ids.shape
        if labels is None:
            logits = self.lm_head(self.model(input_ids))
            loss = None
        else:
            logits = self.lm_head(self.model(input_ids))
            loss = F.cross_entropy(logits.reshape(batch_size * seq_len, -1), labels.reshape(batch_size * seq_len))
        return logits, loss


# ============================================================================
# Scaling to other Qwen3.5 sizes
# ============================================================================

def qwen35_config_from_hf(config: dict) -> Qwen35Config:
    """Map a real Qwen3.5 dense model's config.json into a Qwen35Config.

    Tips: most fields live under `config["text_config"]`; rotary fields are
    nested one level deeper still; `tie_word_embeddings` isn't always where
    you'd expect; one field is spelled differently than in `Qwen35Config`.
    See homework.ipynb for links to real configs.
    """
    # text_config = config["text_config"] (falls back to config itself if absent)
    # pull hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads,
    # head_dim, intermediate_size, vocab_size, rms_norm_eps, hidden_act, layer_types, etc.
    # rope fields (rope_theta, partial_rotary_factor) live under a nested rope_scaling/rope_parameters dict
    # watch for tie_word_embeddings living at the top level, not under text_config
    # build and return Qwen35Config(**mapped_fields)
    if "text_config" in config:
        text_config = config["text_config"]
        mapped_fields = {
            "vocab_size": text_config["vocab_size"],
            "hidden_size": text_config["hidden_size"],
            "intermediate_size": text_config["intermediate_size"],
            "num_hidden_layers": text_config["num_hidden_layers"],
            "num_attention_heads": text_config["num_attention_heads"],
            "num_key_value_heads": text_config["num_key_value_heads"],
            "head_dim": text_config["head_dim"],
            "attention_bias": text_config["attention_bias"],
            "linear_conv_kernel_dim": text_config["linear_conv_kernel_dim"],
            "linear_key_head_dim": text_config["linear_key_head_dim"],
            "linear_value_head_dim": text_config["linear_value_head_dim"],
            "linear_num_key_heads":text_config["linear_num_key_heads"],
            "linear_num_value_heads":text_config["linear_num_value_heads"],
            "rope_theta": text_config["rope_parameters"]["rope_theta"],
            "partial_rotary_factor": text_config["rope_parameters"]["partial_rotary_factor"],
            "max_position_embeddings": text_config["max_position_embeddings"],
            "rms_norm_eps": text_config["rms_norm_eps"],
            "hidden_act": text_config["hidden_act"],
            "tie_word_embeddings": text_config["tie_word_embeddings"] if "tie_word_embeddings" in text_config else config["tie_word_embeddings"],
            "full_attention_interval": text_config["full_attention_interval"],
            "mtp_num_layers":  text_config["mtp_num_hidden_layers"],
        }
        return Qwen35Config(**mapped_fields)
    config["mtp_num_layers"] = config["mtp_num_hidden_layers"]
    config["rope_theta"] = config["rope_parameters"]["rope_theta"]
    config["partial_rotary_factor"] = config["rope_parameters"]["partial_rotary_factor"]
    del config["mtp_num_hidden_layers"], config["rope_parameters"]
    return Qwen35Config(**config)


# ============================================================================
# Bonus tasks
# ============================================================================

# ----------------------------------------------------------------------------
# Bonus: efficient chunked delta rule
# ----------------------------------------------------------------------------

def gated_delta_rule_chunked(query, key, value, g, beta, chunk_size: int = 64):
    """Bonus: chunked delta-rule recurrence -- same result as
    `gated_delta_rule_recurrent` (same input/output shapes), in
    O(seq_len / chunk_size) steps instead of O(seq_len). Hardest block in
    the assignment -- see homework.ipynb for the WY-representation idea.
    """
    # l2norm query/key, fold g's cumulative log-decay and beta into per-chunk terms, same as the recurrent version
    # split the sequence into chunks of chunk_size (pad the last chunk if seq_len isn't a multiple)
    # within a chunk, solve for the WY representation (a lower-triangular system) to get an
    # equivalent parallel/matmul form of the sequential delta-rule update
    # carry `state` across chunks exactly like the recurrent version carries it across timesteps
    raise NotImplementedError("TODO (bonus): implement gated_delta_rule_chunked")


# ----------------------------------------------------------------------------
# Bonus: Multi-Token Prediction (MTP) head
# ----------------------------------------------------------------------------

class Qwen35MTPHead(nn.Module):
    """Bonus: Multi-Token Prediction head. `__init__` is given below. See
    homework.ipynb for the design and tests/test_qwen_bonus.py::TestMTPHead
    for expected shapes referenced by `.forward`.
    """

    def __init__(self, config: Qwen35Config):
        super().__init__()
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        full_attn_config = dataclasses.replace(config, layer_types=["full_attention"] * config.mtp_num_layers)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(full_attn_config, layer_idx) for layer_idx in range(config.mtp_num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35RotaryEmbedding(config)

    def forward(self, hidden_states, input_ids, embed_tokens: nn.Embedding):
        """Predicts input_ids[:, 2:] from hidden_states[:, :-2] and
        input_ids[:, 1:-1]. Concatenate normalized hidden states first, then
        normalized embeddings. Use positions 1..seq_len-2 and a causal mask
        through self.layers, final norm, and embed_tokens.weight.T as the
        classifier. Return logits (batch, seq_len-2, vocab_size); raise
        ValueError for seq_len < 3. This educational head always uses the
        embedding classifier, including when the main LM head is untied.
        See homework.ipynb for the recipe.
        """
        raise NotImplementedError("TODO (bonus): implement Qwen35MTPHead.forward")
