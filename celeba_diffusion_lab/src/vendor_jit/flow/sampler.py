"""Deterministic ODE sampling from ``t=0`` (noise) to ``t=1`` (data).

    x_pred = model(z, t, condition)
    v      = (x_pred - z) / (1 - t).clamp_min(t_eps)
    z_next = z + (t_next - t) * v
    z_next *= valid_mask

Heun is the default; the final step into ``t=1`` falls back to Euler so no second
velocity evaluation happens at the singular endpoint.  Classifier-free guidance
mixes *velocities*, never raw ``x_pred``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import Config

__all__ = ["sample_ode", "velocity_from_x_pred"]


def velocity_from_x_pred(x_pred: Tensor, z: Tensor, t: float | Tensor, t_eps: float) -> Tensor:
    if not isinstance(t, Tensor):
        t = torch.tensor(float(t), device=z.device, dtype=torch.float32)
    denom = (1.0 - t.float()).clamp_min(t_eps)
    return (x_pred.float() - z.float()) / denom


@torch.no_grad()
def sample_ode(
    model,
    cfg: Config,
    valid_mask: Tensor,
    num_samples: int,
    *,
    condition: Tensor | None = None,
    text_embed: Tensor | None = None,
    text_mask: Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    num_steps: int | None = None,
    method: str | None = None,
    cfg_scale: float | None = None,
    cfg_interval: tuple[float, float] | None = None,
    z_init: Tensor | None = None,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    device = torch.device(device)
    shapes = cfg.shapes
    num_steps = cfg.sample.num_steps if num_steps is None else num_steps
    method = (cfg.sample.method if method is None else method).lower()
    cfg_scale = cfg.sample.cfg_scale if cfg_scale is None else cfg_scale
    cfg_interval = tuple(cfg.sample.cfg_interval if cfg_interval is None else cfg_interval)
    if method not in {"euler", "heun"}:
        raise ValueError(f"unsupported sampler {method!r}")

    mask = valid_mask.to(device=device, dtype=dtype)
    if z_init is None:
        z = torch.randn(
            num_samples,
            cfg.model.in_channels,
            shapes.freq_pad,
            shapes.time_pad,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        z = z * cfg.flow.noise_scale * mask
    else:
        z = (z_init.to(device=device, dtype=dtype) * mask).clone()

    inner = getattr(model, "module", model)
    model_cfg = getattr(inner, "cfg", cfg.model)
    has_text = text_embed is not None and getattr(inner, "text_proj", None) is not None
    has_class = condition is not None and model_cfg.num_classes > 0
    use_cfg = (has_text or has_class) and cfg_scale != 1.0
    if condition is not None:
        condition = condition.to(device=device).reshape(-1)
        if condition.shape[0] != num_samples:
            raise ValueError("condition must have one entry per sample")
    if text_embed is not None:
        text_embed = text_embed.to(device=device, dtype=dtype)
        if text_embed.shape[0] != num_samples:
            raise ValueError("text_embed must have one entry per sample")
        text_mask = (
            text_mask.to(device=device)
            if text_mask is not None
            else torch.ones(text_embed.shape[:2], dtype=torch.bool, device=device)
        )

    t_eps = cfg.flow.t_eps
    grid = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64).tolist()
    trajectory: list[Tensor] = []

    def velocity(z_in: Tensor, t_scalar: float) -> Tensor:
        t_batch = torch.full((z_in.shape[0],), float(t_scalar), device=device, dtype=torch.float32)
        in_interval = cfg_interval[0] <= t_scalar <= cfg_interval[1]
        if not use_cfg or not in_interval:
            x_pred = model(z_in, t_batch, condition, text_embed=text_embed, text_mask=text_mask)
            return velocity_from_x_pred(x_pred, z_in, t_scalar, t_eps).to(dtype) * mask
        # one doubled-batch forward; numerically identical to two separate passes.
        # class and text are dropped together for the unconditional half, so CFG
        # mixes a single coherent "conditional vs unconditional" pair of velocities.
        b = z_in.shape[0]
        z_pair = torch.cat([z_in, z_in], dim=0)
        t_pair = torch.cat([t_batch, t_batch], dim=0)
        drop = torch.zeros(2 * b, dtype=torch.bool, device=device)
        drop[b:] = True
        cond_pair = torch.cat([condition, condition], dim=0) if condition is not None else None
        text_pair = torch.cat([text_embed, text_embed], dim=0) if text_embed is not None else None
        text_mask_pair = torch.cat([text_mask, text_mask], dim=0) if text_mask is not None else None
        x_pred = model(
            z_pair,
            t_pair,
            cond_pair,
            text_embed=text_pair,
            text_mask=text_mask_pair,
            force_drop_mask=drop,
            force_drop_text_mask=drop,
        )
        x_cond, x_uncond = x_pred[:b], x_pred[b:]
        v_cond = velocity_from_x_pred(x_cond, z_in, t_scalar, t_eps)
        v_uncond = velocity_from_x_pred(x_uncond, z_in, t_scalar, t_eps)
        return (v_uncond + cfg_scale * (v_cond - v_uncond)).to(dtype) * mask

    for i in range(num_steps):
        t_cur, t_next = float(grid[i]), float(grid[i + 1])
        dt = t_next - t_cur
        v = velocity(z, t_cur)
        if method == "heun" and t_next < 1.0:
            z_euler = (z + dt * v) * mask
            v2 = velocity(z_euler, t_next)
            v = 0.5 * (v + v2)
        z = (z + dt * v) * mask
        if return_trajectory:
            trajectory.append(z.clone())

    if return_trajectory:
        return z, trajectory
    return z
