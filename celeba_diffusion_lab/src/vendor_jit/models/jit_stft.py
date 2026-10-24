"""JiTSTFT: a plain transformer that maps a noisy Re/Im STFT to the clean one.

``x_pred = model(z_t, t, condition)`` with ``z_t``/``x_pred`` of shape
``[B, 2, F_pad, T_pad]``.  The network output *is* the clean signal estimate: no
EDM preconditioner, no input-to-output skip, no tokenizer, no U-Net. An optional
small conv stack (:class:`OutputConv`, off by default) may follow unpatchify to
mix across patch boundaries in pixel/spectrogram space -- see its docstring;
that is output-to-output refinement, not a skip from the noisy input ``z_t``.
An optional continuous vector conditioning pathway (``attr_condition``, gated
by ``ModelConfig.attr_cond_dim``, 0/off by default) sums a small MLP embedding
into the same adaLN ``cond`` vector as ``class_embed``, independent of it --
zero-initialized, so it is a no-op until trained.

Everything is rectangular.  Frequency and time are distinct axes with distinct
patch sizes, distinct grid extents and distinct positional frequencies; nothing
in this file may assume a square image or ``sqrt(num_patches)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import Config, ModelConfig, ShapeSpec, TextConditionConfig

__all__ = ["JiTSTFT", "build_model", "MODEL_PRESETS", "CrossAttention"]

# depth / hidden_dim / num_heads / bottleneck_dim, per AGENTS.md 6.3
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"depth": 2, "hidden_dim": 128, "num_heads": 4, "bottleneck_dim": 32},
    "small": {"depth": 8, "hidden_dim": 512, "num_heads": 8, "bottleneck_dim": 96},
    "base": {"depth": 12, "hidden_dim": 768, "num_heads": 12, "bottleneck_dim": 128},
}


# --------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x32.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t: Tensor, dim: int, scale: float, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal embedding of a continuous ``t`` in ``[0, 1]``."""
    if dim % 2 != 0:
        raise ValueError("timestep embedding dim must be even")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().reshape(-1, 1) * scale * freqs.reshape(1, -1)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def build_2d_sincos_position_embedding(
    grid_freq: int, grid_time: int, dim: int, temperature: float = 10000.0
) -> Tensor:
    """Fixed rectangular 2D sin-cos table, ``[1, grid_freq * grid_time, dim]``.

    The frequency axis and the time axis each own half of the channels; the two
    grid extents are independent, so non-square grids are the normal case.
    """
    if dim % 4 != 0:
        raise ValueError("2D sin-cos position embedding requires dim divisible by 4")
    quarter = dim // 4
    omega = 1.0 / (temperature ** (torch.arange(quarter, dtype=torch.float64) / quarter))
    pos_f = torch.arange(grid_freq, dtype=torch.float64)
    pos_t = torch.arange(grid_time, dtype=torch.float64)
    # token order is frequency-major, matching patchify()
    grid_f = pos_f.reshape(-1, 1).expand(grid_freq, grid_time).reshape(-1)
    grid_t = pos_t.reshape(1, -1).expand(grid_freq, grid_time).reshape(-1)
    out_f = grid_f.reshape(-1, 1) * omega.reshape(1, -1)
    out_t = grid_t.reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([out_f.sin(), out_f.cos(), out_t.sin(), out_t.cos()], dim=1)  # [N, dim]
    return emb.to(torch.float32).unsqueeze(0)


def build_2d_rope_tables(
    grid_freq: int, grid_time: int, head_dim: int, base: float = 10000.0
) -> tuple[Tensor, Tensor]:
    """Rectangular 2D RoPE: the rotation pairs of every head are split between axes.

    Returns ``(cos, sin)`` of shape ``[N, head_dim // 2]`` where the first quarter
    of the head dimension rotates with the frequency index and the second with
    the time index.
    """
    if head_dim % 4 != 0:
        raise ValueError("rectangular 2D RoPE requires head_dim divisible by 4")
    axis_dim = head_dim // 2
    pairs = axis_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(pairs, dtype=torch.float64) * 2.0 / axis_dim))
    pos_f = torch.arange(grid_freq, dtype=torch.float64).reshape(-1, 1).expand(grid_freq, grid_time)
    pos_t = torch.arange(grid_time, dtype=torch.float64).reshape(1, -1).expand(grid_freq, grid_time)
    ang_f = pos_f.reshape(-1, 1) * inv_freq.reshape(1, -1)
    ang_t = pos_t.reshape(-1, 1) * inv_freq.reshape(1, -1)
    ang = torch.cat([ang_f, ang_t], dim=-1)  # [N, head_dim // 2]
    return ang.cos().to(torch.float32), ang.sin().to(torch.float32)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """``x``: ``[B, H, N, Dh]``; ``cos``/``sin``: ``[N, Dh // 2]``."""
    dtype = x.dtype
    x32 = x.float().reshape(*x.shape[:-1], -1, 2)
    x_re, x_im = x32[..., 0], x32[..., 1]
    cos = cos.reshape(1, 1, *cos.shape)
    sin = sin.reshape(1, 1, *sin.shape)
    out = torch.stack([x_re * cos - x_im * sin, x_re * sin + x_im * cos], dim=-1)
    return out.flatten(-2).to(dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Image tokens (query) attend to a frozen text encoder's projected sequence."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor | None) -> Tensor:
        b, n, c = x.shape
        m = context.shape[1]
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(b, m, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_mask = None
        if context_mask is not None:
            # bool mask: True = attend, broadcasts over heads and query positions
            attn_mask = context_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=True)
        self.w3 = nn.Linear(dim, ffn_dim, bias=True)
        self.w2 = nn.Linear(ffn_dim, dim, bias=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        if use_cross_attn:
            self.norm_c = RMSNorm(dim, elementwise_affine=False)
            self.cross_attn = CrossAttention(dim, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        self.mlp = SwiGLU(dim, ffn_dim, dropout=dropout)
        num_mod = 9 if use_cross_attn else 6
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, num_mod * dim, bias=True))

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        rope: tuple[Tensor, Tensor] | None,
        text_tokens: Tensor | None = None,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        if self.use_cross_attn:
            shift_a, scale_a, gate_a, shift_c, scale_c, gate_c, shift_m, scale_m, gate_m = (
                self.ada_ln(cond).chunk(9, dim=-1)
            )
        else:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_ln(cond).chunk(6, dim=-1)
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_a, scale_a), rope)
        if self.use_cross_attn:
            x = x + gate_c.unsqueeze(1) * self.cross_attn(
                modulate(self.norm_c(x), shift_c, scale_c), text_tokens, text_mask
            )
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class OutputConv(nn.Module):
    """Optional post-unpatchify pixel-space refinement, gated by
    ``ModelConfig.output_conv_layers`` (0 = module absent, the default).

    A patchified transformer with no spatial mixing after unpatchify predicts
    every patch independently, so early in training it captures global color/
    layout statistics but not cross-patch texture continuity (visible as a
    patch-grid artifact -- see docs/execplans/jit_celeba.md, following the
    "Patch Detailer Head" idea from arXiv:2511.18822, simplified to a small
    conv stack over the assembled ``[B, C, H, W]`` grid instead of a per-patch
    U-Net with a global context vector). The last conv is zero-initialized so
    this starts as an exact identity (``x + 0 == x``): with the rest of the
    network's existing adaLN-Zero init making the *base* output zero too, the
    ``test_output_is_zero_at_initialisation`` invariant holds whether or not
    this head is enabled.
    """

    def __init__(self, channels: int, *, hidden: int, num_layers: int, kernel_size: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        c_in = channels
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(c_in, hidden, kernel_size, padding=pad),
                nn.GroupNorm(min(8, hidden), hidden),
                nn.SiLU(),
            ]
            c_in = hidden
        layers.append(nn.Conv2d(c_in, channels, kernel_size, padding=pad))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.proj = nn.Linear(dim, patch_dim, bias=True)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.ada_ln(cond).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


# --------------------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------------------
class JiTSTFT(nn.Module):
    def __init__(
        self,
        model_cfg: ModelConfig,
        shapes: ShapeSpec,
        text_cfg: TextConditionConfig | None = None,
    ) -> None:
        super().__init__()
        model_cfg.validate()
        self.cfg = model_cfg
        self.shapes = shapes
        self.text_cfg = text_cfg
        dim = model_cfg.hidden_dim
        self.head_dim = dim // model_cfg.num_heads
        use_cross_attn = text_cfg is not None

        if use_cross_attn:
            # frozen text-encoder hidden size -> this model's hidden_dim, plus a
            # single learned "null text" token substituted for CFG/unconditional drops
            self.text_proj = nn.Sequential(
                nn.Linear(text_cfg.encoder_hidden_size, dim, bias=True),
                RMSNorm(dim),
            )
            self.null_text = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.text_proj = None
            self.null_text = None

        # low-rank (bottleneck) linear patch embedding; no convolutional acoustic front-end
        self.patch_proj = nn.Conv2d(
            model_cfg.in_channels,
            model_cfg.bottleneck_dim,
            kernel_size=(shapes.patch_freq, shapes.patch_time),
            stride=(shapes.patch_freq, shapes.patch_time),
            bias=False,
        )
        self.patch_expand = nn.Conv2d(model_cfg.bottleneck_dim, dim, kernel_size=1, bias=True)

        self.t_embed = nn.Sequential(
            nn.Linear(dim, dim, bias=True), nn.SiLU(), nn.Linear(dim, dim, bias=True)
        )
        self.class_embed = (
            nn.Embedding(model_cfg.num_classes + 1, dim) if model_cfg.num_classes > 0 else None
        )
        self.attr_embed = (
            nn.Sequential(
                nn.Linear(model_cfg.attr_cond_dim, dim, bias=True),
                nn.SiLU(),
                nn.Linear(dim, dim, bias=True),
            )
            if model_cfg.attr_cond_dim > 0
            else None
        )
        self.cond_tokens = (
            nn.Parameter(torch.zeros(1, model_cfg.num_cond_tokens, dim))
            if model_cfg.num_cond_tokens > 0
            else None
        )

        if model_cfg.use_sincos_pos:
            self.register_buffer(
                "pos_embed",
                build_2d_sincos_position_embedding(shapes.grid_freq, shapes.grid_time, dim),
                persistent=False,
            )
        else:
            self.pos_embed = None
        if model_cfg.use_rope:
            cos, sin = build_2d_rope_tables(
                shapes.grid_freq, shapes.grid_time, self.head_dim, model_cfg.rope_base
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

        ffn_dim = _round_to(int(2 * model_cfg.mlp_ratio * dim / 3), model_cfg.ffn_multiple_of)
        depth = model_cfg.depth
        mid_lo, mid_hi = depth // 4, depth - depth // 4
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    model_cfg.num_heads,
                    ffn_dim,
                    dropout=model_cfg.dropout if mid_lo <= i < mid_hi else 0.0,
                    use_cross_attn=use_cross_attn,
                )
                for i in range(depth)
            ]
        )
        self.final = FinalLayer(dim, shapes.patch_dim)
        self.output_conv = (
            OutputConv(
                model_cfg.in_channels,
                hidden=model_cfg.output_conv_hidden,
                num_layers=model_cfg.output_conv_layers,
                kernel_size=model_cfg.output_conv_kernel,
            )
            if model_cfg.output_conv_layers > 0
            else None
        )
        self.initialize_weights()

    # ------------------------------------------------------------------- init
    def initialize_weights(self) -> None:
        def _basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic)
        for conv in (self.patch_proj, self.patch_expand):
            nn.init.xavier_uniform_(conv.weight.view(conv.weight.shape[0], -1))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)
        if self.class_embed is not None:
            nn.init.normal_(self.class_embed.weight, std=0.02)
        if self.attr_embed is not None:
            nn.init.normal_(self.attr_embed[0].weight, std=0.02)
            nn.init.zeros_(self.attr_embed[0].bias)
            # zero-init the last layer: attr_embed(x) == 0 for any x at init,
            # so attr_cond_dim>0 configs start behaviorally unconditional
            # (same convention as adaLN-Zero / OutputConv elsewhere here).
            nn.init.zeros_(self.attr_embed[2].weight)
            nn.init.zeros_(self.attr_embed[2].bias)
        if self.cond_tokens is not None:
            nn.init.normal_(self.cond_tokens, std=0.02)
        if self.text_proj is not None:
            nn.init.normal_(self.text_proj[0].weight, std=0.02)
            nn.init.zeros_(self.text_proj[0].bias)
            nn.init.normal_(self.null_text, std=0.02)
        # adaLN-Zero: every modulation head and the output projection start at zero
        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[-1].weight)
            nn.init.zeros_(block.ada_ln[-1].bias)
        nn.init.zeros_(self.final.ada_ln[-1].weight)
        nn.init.zeros_(self.final.ada_ln[-1].bias)
        nn.init.zeros_(self.final.proj.weight)
        nn.init.zeros_(self.final.proj.bias)
        if self.output_conv is not None:
            convs = [m for m in self.output_conv.net if isinstance(m, nn.Conv2d)]
            for conv in convs[:-1]:
                nn.init.xavier_uniform_(conv.weight.view(conv.weight.shape[0], -1))
                nn.init.zeros_(conv.bias)
            # last conv zero-init: output_conv(x) == x at init, whatever x is
            nn.init.zeros_(convs[-1].weight)
            nn.init.zeros_(convs[-1].bias)

    # ------------------------------------------------------- patch <-> tokens
    def patchify(self, x: Tensor) -> Tensor:
        """``[B, 2, F_pad, T_pad]`` -> ``[B, N, 2 * p_f * p_t]`` (frequency-major)."""
        s = self.shapes
        b, c, fpad, tpad = x.shape
        if (fpad, tpad) != (s.freq_pad, s.time_pad):
            raise ValueError(f"expected [..., {s.freq_pad}, {s.time_pad}], got {(fpad, tpad)}")
        x = x.reshape(b, c, s.grid_freq, s.patch_freq, s.grid_time, s.patch_time)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, gf, gt, C, pf, pt]
        return x.reshape(b, s.num_patches, s.patch_dim)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        """``[B, N, 2 * p_f * p_t]`` -> ``[B, 2, F_pad, T_pad]``."""
        s = self.shapes
        b = tokens.shape[0]
        if tokens.shape[1:] != (s.num_patches, s.patch_dim):
            raise ValueError(
                f"expected [B, {s.num_patches}, {s.patch_dim}], got {tuple(tokens.shape)}"
            )
        x = tokens.reshape(
            b, s.grid_freq, s.grid_time, self.cfg.in_channels, s.patch_freq, s.patch_time
        )
        x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, gf, pf, gt, pt]
        return x.reshape(b, self.cfg.in_channels, s.freq_pad, s.time_pad)

    # ---------------------------------------------------------------- forward
    def embed_condition(
        self,
        t: Tensor,
        condition: Tensor | None,
        *,
        force_drop_mask: Tensor | None = None,
        attr_condition: Tensor | None = None,
        force_drop_attr_mask: Tensor | None = None,
    ) -> Tensor:
        cond = self.t_embed(
            timestep_embedding(t, self.cfg.hidden_dim, self.cfg.time_embed_scale).to(
                self.t_embed[0].weight.dtype
            )
        )
        if self.class_embed is not None:
            null_index = self.cfg.num_classes
            if condition is None:
                labels = torch.full((t.shape[0],), null_index, dtype=torch.long, device=t.device)
            else:
                labels = condition.to(device=t.device, dtype=torch.long).reshape(-1)
                if labels.shape[0] != t.shape[0]:
                    raise ValueError("condition batch does not match t batch")
                if force_drop_mask is not None:
                    drop = force_drop_mask.to(device=labels.device, dtype=torch.bool).reshape(-1)
                    labels = torch.where(drop, torch.full_like(labels, null_index), labels)
                elif self.training and self.cfg.class_dropout_prob > 0.0:
                    drop = (
                        torch.rand(labels.shape, device=labels.device) < self.cfg.class_dropout_prob
                    )
                    labels = torch.where(drop, torch.full_like(labels, null_index), labels)
            cond = cond + self.class_embed(labels)
        if self.attr_embed is not None:
            if attr_condition is None:
                raise ValueError("attr_condition is required when attr_cond_dim > 0")
            attr_condition = attr_condition.to(device=t.device, dtype=cond.dtype)
            if attr_condition.shape != (t.shape[0], self.cfg.attr_cond_dim):
                raise ValueError(
                    f"expected attr_condition [B={t.shape[0]}, {self.cfg.attr_cond_dim}], "
                    f"got {tuple(attr_condition.shape)}"
                )
            if force_drop_attr_mask is not None:
                drop = force_drop_attr_mask.to(device=t.device, dtype=cond.dtype).reshape(-1, 1)
                attr_condition = attr_condition * (1.0 - drop)
            elif self.training and self.cfg.attr_dropout_prob > 0.0:
                drop = (
                    (torch.rand(t.shape[0], device=t.device) < self.cfg.attr_dropout_prob)
                    .to(cond.dtype)
                    .reshape(-1, 1)
                )
                attr_condition = attr_condition * (1.0 - drop)
            cond = cond + self.attr_embed(attr_condition)
        return cond

    def embed_text(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text_embed: Tensor | None,
        text_mask: Tensor | None,
        *,
        force_drop_text_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Project the frozen text-encoder sequence; substitute the learned null
        token for CFG-dropped rows (or the whole batch, when ``text_embed is None``).
        """
        if text_embed is None:
            tokens = self.null_text.to(dtype).expand(batch_size, 1, -1)
            mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            return tokens, mask

        tokens = self.text_proj(text_embed.to(dtype))
        if tokens.shape[0] != batch_size:
            raise ValueError("text_embed batch does not match z_t batch")
        mask = (
            text_mask.to(device=tokens.device, dtype=torch.bool)
            if text_mask is not None
            else torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        )

        if force_drop_text_mask is not None:
            drop = force_drop_text_mask.to(device=tokens.device, dtype=torch.bool).reshape(-1)
        elif self.training and self.text_cfg.drop_prob > 0.0:
            drop = torch.rand(batch_size, device=tokens.device) < self.text_cfg.drop_prob
        else:
            drop = None
        if drop is not None:
            # Always blend in null_text (multiplied by a 0/1 mask) rather than only
            # touching it when `drop.any()`: under DDP, a parameter that enters the
            # autograd graph on some iterations (batches with >=1 dropped row) and
            # not others (batches with none, which happens ~3% of the time at
            # micro_batch=32/drop_prob=0.1) breaks the static-graph assumption
            # ("Expected to have finished reduction in the prior iteration...").
            # An always-executed multiply-by-zero keeps null_text in the graph on
            # every call, with the same numerical result as the conditional version.
            drop_f = drop.to(tokens.dtype).reshape(-1, 1)
            null = self.null_text[0, 0].to(tokens.dtype)
            new_first = tokens[:, 0] * (1.0 - drop_f) + null.unsqueeze(0) * drop_f
            tokens = torch.cat([new_first.unsqueeze(1), tokens[:, 1:]], dim=1)
            mask = mask.clone()
            mask[:, 0] = True
            mask[:, 1:] = mask[:, 1:] & ~drop.reshape(-1, 1)
        return tokens, mask

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        condition: Tensor | None = None,
        text_embed: Tensor | None = None,
        text_mask: Tensor | None = None,
        *,
        force_drop_mask: Tensor | None = None,
        force_drop_text_mask: Tensor | None = None,
        attr_condition: Tensor | None = None,
        force_drop_attr_mask: Tensor | None = None,
    ) -> Tensor:
        if z_t.dim() != 4:
            raise ValueError(f"expected [B, 2, F_pad, T_pad], got {tuple(z_t.shape)}")
        t = t.reshape(-1)
        if t.shape[0] != z_t.shape[0]:
            raise ValueError("t must have one entry per batch element")

        h = self.patch_expand(self.patch_proj(z_t))  # [B, D, gf, gt]
        h = h.flatten(2).transpose(1, 2)  # [B, N, D] frequency-major
        if self.pos_embed is not None:
            h = h + self.pos_embed.to(h.dtype)

        rope = None
        if self.rope_cos is not None:
            rope = (self.rope_cos.to(h.dtype), self.rope_sin.to(h.dtype))

        num_prefix = 0
        if self.cond_tokens is not None:
            prefix = self.cond_tokens.to(h.dtype).expand(h.shape[0], -1, -1)
            h = torch.cat([prefix, h], dim=1)
            num_prefix = prefix.shape[1]
            if rope is not None:
                pad_cos = torch.ones(num_prefix, rope[0].shape[1], device=h.device, dtype=h.dtype)
                pad_sin = torch.zeros_like(pad_cos)
                rope = (torch.cat([pad_cos, rope[0]]), torch.cat([pad_sin, rope[1]]))

        text_tokens = text_kv_mask = None
        if self.text_proj is not None:
            text_tokens, text_kv_mask = self.embed_text(
                h.shape[0],
                h.device,
                h.dtype,
                text_embed,
                text_mask,
                force_drop_text_mask=force_drop_text_mask,
            )

        cond = self.embed_condition(
            t,
            condition,
            force_drop_mask=force_drop_mask,
            attr_condition=attr_condition,
            force_drop_attr_mask=force_drop_attr_mask,
        ).to(h.dtype)
        for block in self.blocks:
            h = block(h, cond, rope, text_tokens=text_tokens, text_mask=text_kv_mask)
        h = self.final(h, cond)
        if num_prefix:
            h = h[:, num_prefix:]
        out = self.unpatchify(h)
        if self.output_conv is not None:
            out = self.output_conv(out)
        return out


def _round_to(value: int, multiple: int) -> int:
    return max(multiple, int(math.ceil(value / multiple) * multiple))


def build_model(cfg: Config) -> JiTSTFT:
    text_cfg = cfg.text if cfg.text.enabled else None
    return JiTSTFT(cfg.model, cfg.shapes, text_cfg=text_cfg)
