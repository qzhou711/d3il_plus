"""Discrete-time consistency model for distilling a frozen DDPM teacher.

The wrapper mirrors :class:`agents.models.diffusion.gc_diffusion.Diffusion`
just enough so that :class:`agents.cd_agent.ConsistencyDistillationAgent`
can plug into the existing :class:`agents.base_agent.BaseAgent` pipeline
without touching the simulation / dataset layers.

Key design choices:
- discrete schedule reused from the teacher (cosine, n_timesteps=8);
- student backbone has the same architecture as the teacher (warm-start);
- student parameterisation is selectable via ``student_param``;
- consistency loss supports ``"pseudo_huber"`` (default) and ``"l2"``;
- target network is a separate EMA copy maintained outside this module;
- ``f_theta`` does NOT clamp its output during training so that the CM
  boundary condition ``f_theta(x, 0) = x`` is respected; clipping is only
  applied to ``x0_hat`` (the data-space prediction) and at the end of
  ``sample()`` for the action that hits the simulator.
"""

from typing import Optional, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .utils import (
    cosine_beta_schedule,
    extract,
    linear_beta_schedule,
    vp_beta_schedule,
)


class ConsistencyDiffusion(nn.Module):
    """Consistency model wrapping a teacher-style ε / x0 backbone."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model: DictConfig,
        beta_schedule: str,
        n_timesteps: int,
        loss_type: str,
        clip_denoised: bool,
        predict_epsilon: bool = True,
        device: str = "cuda",
        student_param: str = "eps",
        huber_c: float = 7.6e-4,
        sigma_data: float = 1.0,
        boundary_scaling: str = "edm",
        lambda_direct: float = 0.0,
        weight_by_sigma: bool = False,
    ) -> None:
        super().__init__()
        if student_param not in {"eps", "x0"}:
            raise ValueError(f"student_param must be 'eps' or 'x0', got {student_param!r}")
        if loss_type not in {"pseudo_huber", "l2"}:
            raise ValueError(f"loss_type must be 'pseudo_huber' or 'l2', got {loss_type!r}")
        if boundary_scaling not in {"edm", "sqrt_alpha_bar", "identity"}:
            raise ValueError(
                "boundary_scaling must be 'edm', 'sqrt_alpha_bar' or 'identity', "
                f"got {boundary_scaling!r}"
            )

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Set externally by the agent right after instantiation.
        self.min_action: Optional[torch.Tensor] = None
        self.max_action: Optional[torch.Tensor] = None
        self.predict_epsilon = predict_epsilon
        self.student_param = student_param
        self.loss_type = loss_type
        self.huber_c = huber_c
        self.clip_denoised = clip_denoised
        self.sigma_data = sigma_data
        self.boundary_scaling = boundary_scaling
        self.lambda_direct = float(lambda_direct)
        self.weight_by_sigma = bool(weight_by_sigma)

        if beta_schedule == "linear":
            self.betas = linear_beta_schedule(n_timesteps).to(device)
        elif beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(n_timesteps).to(device)
        elif beta_schedule == "vp":
            self.betas = vp_beta_schedule(n_timesteps).to(device)
        else:
            raise ValueError(f"unsupported beta_schedule {beta_schedule!r}")

        self.n_timesteps = n_timesteps

        self.model = hydra.utils.instantiate(model)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1).to(device)

        # Karras/EDM-style preconditioning coefficients with sigma_min := sigma(t=0)
        # so that ``f_theta(x, 0) = x`` exactly.
        sigma = torch.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod)
        self.sigma_t = sigma.to(device)
        self.sigma_min = sigma[0].to(device)
        sd2 = self.sigma_data ** 2
        delta = self.sigma_t - self.sigma_min
        self.c_skip_edm = sd2 / (delta * delta + sd2)
        self.c_out_edm = self.sigma_data * delta / torch.sqrt(self.sigma_t * self.sigma_t + sd2)

    # ------------------------------------------------------------------
    # parameter helpers
    # ------------------------------------------------------------------
    def get_params(self):
        return self.model.get_params()

    # ------------------------------------------------------------------
    # forward / inverse diffusion helpers
    # ------------------------------------------------------------------
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_x0_from_eps(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * eps
        )

    def predict_eps_from_x0(self, x: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

    # ------------------------------------------------------------------
    # consistency parameterisation f_theta(x, t, s, g)
    # ------------------------------------------------------------------
    def _backbone_x0(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.model(x, t, s, g)
        if self.student_param == "eps":
            x0_hat = self.predict_x0_from_eps(x, t, out)
        else:
            x0_hat = out
        # Clip x0_hat in data space; the consistency output ``c_skip*x + c_out*x0_hat``
        # is NOT clipped here (would break the CM boundary condition).
        if self.clip_denoised and self.min_action is not None and self.max_action is not None:
            x0_hat = torch.clamp(x0_hat, self.min_action, self.max_action)
        return x0_hat

    def _boundary_coefs(self, x_shape, t: torch.Tensor):
        if self.boundary_scaling == "edm":
            c_skip = extract(self.c_skip_edm, t, x_shape)
            c_out = extract(self.c_out_edm, t, x_shape)
        elif self.boundary_scaling == "sqrt_alpha_bar":
            c_skip = extract(self.sqrt_alphas_cumprod, t, x_shape)
            c_out = extract(self.sqrt_one_minus_alphas_cumprod, t, x_shape)
        else:  # identity
            c_skip = torch.zeros((), device=self.device).expand(x_shape[0], *([1] * (len(x_shape) - 1)))
            c_out = torch.ones((), device=self.device).expand(x_shape[0], *([1] * (len(x_shape) - 1)))
        return c_skip, c_out

    def f_theta(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Consistency mapping ``f_theta(x_t, t) -> x_0`` with skip connection.

        No output clipping here: ``c_skip*x`` may legitimately fall outside the
        action bounds (the input ``x`` is noisy or pure Gaussian) and the CM
        boundary condition requires identity at ``t=0``. The data-space
        prediction ``x0_hat`` is clipped inside :meth:`_backbone_x0`.
        """
        x0_hat = self._backbone_x0(x, t, s, g)
        c_skip, c_out = self._boundary_coefs(x.shape, t)
        return c_skip * x + c_out * x0_hat

    # ------------------------------------------------------------------
    # consistency distillation training step
    # ------------------------------------------------------------------
    def _ddim_one_step(
        self,
        x_next: torch.Tensor,
        t_next: torch.Tensor,
        t_curr: torch.Tensor,
        s: torch.Tensor,
        g: Optional[torch.Tensor],
        teacher: nn.Module,
    ) -> torch.Tensor:
        """Deterministic DDIM step using the frozen teacher's ε predictor."""
        with torch.no_grad():
            eps_teacher = teacher.model(x_next, t_next, s, g)
            x0_teacher = self.predict_x0_from_eps(x_next, t_next, eps_teacher)
            if self.clip_denoised and self.min_action is not None and self.max_action is not None:
                x0_teacher = torch.clamp(x0_teacher, self.min_action, self.max_action)
            sqrt_acp_curr = extract(self.sqrt_alphas_cumprod, t_curr, x_next.shape)
            sqrt_1macp_curr = extract(self.sqrt_one_minus_alphas_cumprod, t_curr, x_next.shape)
            x_curr = sqrt_acp_curr * x0_teacher + sqrt_1macp_curr * eps_teacher
        return x_curr

    def _consistency_distance(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.loss_type == "l2":
            per = (pred - target) ** 2
        else:
            diff = pred - target
            per = torch.sqrt(diff * diff + self.huber_c * self.huber_c) - self.huber_c
        if weight is None:
            return per.mean()
        # Per-sample weight broadcasted over feature dims.
        if per.dim() == 3:
            weight = weight.view(-1, 1, 1)
        else:
            weight = weight.view(-1, 1)
        return (per * weight).mean()

    def cd_loss_components(
        self,
        x_start: torch.Tensor,
        state: torch.Tensor,
        teacher: nn.Module,
        target_net: "ConsistencyDiffusion",
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (consistency_term, anchor_term)."""
        batch_size = x_start.shape[0]
        n = torch.randint(0, self.n_timesteps - 1, (batch_size,), device=x_start.device, dtype=torch.long)
        t_curr = n
        t_next = n + 1

        noise = torch.randn_like(x_start)
        x_next = self.q_sample(x_start=x_start, t=t_next, noise=noise)

        x_curr = self._ddim_one_step(x_next, t_next, t_curr, state, goal, teacher)

        pred = self.f_theta(x_next, t_next, state, goal)
        with torch.no_grad():
            target = target_net.f_theta(x_curr, t_curr, state, goal)

        # Optional sigma-weighting: emphasise low-noise (small t) where the
        # consistency constraint should be tight, de-emphasise high-noise where
        # ``x_start`` is essentially Gaussian and the target is noisy. Disabled
        # by default to keep the pure CM formulation.
        if self.weight_by_sigma:
            weight = 1.0 / (self.sigma_t[t_next] + 1.0)
            cd_term = self._consistency_distance(pred, target, weight=weight)
        else:
            cd_term = self._consistency_distance(pred, target)

        if self.lambda_direct > 0.0:
            t_top = torch.full(
                (batch_size,), self.n_timesteps - 1,
                device=x_start.device, dtype=torch.long,
            )
            x_top = torch.randn_like(x_start)
            pred_top = self.f_theta(x_top, t_top, state, goal)
            anchor_term = self._consistency_distance(pred_top, x_start)
        else:
            anchor_term = torch.zeros((), device=x_start.device)

        return cd_term, anchor_term

    def cd_loss(
        self,
        x_start: torch.Tensor,
        state: torch.Tensor,
        teacher: nn.Module,
        target_net: "ConsistencyDiffusion",
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cd_term, anchor_term = self.cd_loss_components(x_start, state, teacher, target_net, goal)
        return cd_term + self.lambda_direct * anchor_term

    # ------------------------------------------------------------------
    # consistency model sampling (1-step or multi-step)
    # ------------------------------------------------------------------
    def _build_inference_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Pick a descending sub-sequence of original timesteps for K-step sampling.

        For small ``N`` (e.g. 8), the most informative iterates lie in the
        upper half of the schedule. We use a head-K sub-sequence
        ``[T-1, T-2, ..., T-K]`` so K-step sampling progressively re-uses the
        finer time grid. Falls back to ``[T-1]`` for K=1.
        """
        T = self.n_timesteps
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        K = min(num_inference_steps, T - 1)
        steps = [T - 1 - k for k in range(K)]
        return torch.tensor(steps, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        num_inference_steps: int = 1,
        return_diffusion: bool = False,
    ) -> torch.Tensor:
        """Multi-step consistency sampling.

        Algorithm 1 from the consistency-models paper, adapted to a discrete
        schedule. Run ``f_theta`` at the largest timestep, then alternate
        re-noising via ``q_sample(x0_hat, t)`` and ``f_theta`` for the
        remaining timesteps.
        """
        batch_size = state.shape[0]
        if state.dim() == 3:
            shape = (batch_size, state.shape[1], self.action_dim)
        else:
            shape = (batch_size, self.action_dim)

        timesteps = self._build_inference_timesteps(num_inference_steps)
        x = torch.randn(shape, device=self.device)

        if return_diffusion:
            chain = [x]

        first_t = int(timesteps[0].item())
        t_batch = torch.full((batch_size,), first_t, device=self.device, dtype=torch.long)
        x = self.f_theta(x, t_batch, state, goal)
        if return_diffusion:
            chain.append(x)

        for t in timesteps[1:]:
            t_int = int(t.item())
            t_batch = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)
            # Re-noise the current x0 estimate to noise level t, then apply f_theta.
            noise = torch.randn_like(x)
            x_t = (
                extract(self.sqrt_alphas_cumprod, t_batch, x.shape) * x
                + extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape) * noise
            )
            x = self.f_theta(x_t, t_batch, state, goal)
            if return_diffusion:
                chain.append(x)

        if self.min_action is not None and self.max_action is not None:
            x = torch.clamp(x, self.min_action, self.max_action)

        if return_diffusion:
            return x, torch.stack(chain, dim=1)
        return x

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        num_inference_steps: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        return self.sample(state, goal, num_inference_steps=num_inference_steps, **kwargs)
