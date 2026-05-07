"""Consistency-distillation agent.

Mirrors the surface area of :class:`agents.ddpm_agent.DiffusionAgent` so that
the existing simulation, dataset, and evaluation code can drive a distilled
student without modification. Differences vs the teacher agent:

- Maintains a **frozen teacher** loaded from ``teacher_ckpt``.
- Maintains a **target network** (deep-copy of the student) updated by EMA
  with ``target_decay``; this is the ``f_theta_minus`` in CD literature.
- Training step calls
  :meth:`agents.models.diffusion.consistency.ConsistencyDiffusion.cd_loss_components`.
- ``predict`` performs K-step consistency sampling instead of full DDPM
  reverse process.

The previous version maintained a **second** EMA copy of the student for
inference, while the target network tracked the *raw* training weights.
That meant ``predict``/``evaluate``/``store_model_weights`` operated on a
different parameterisation than the one being optimised, which contributed
to the train-eval divergence we observed in the first round of experiments.
This file removes the inference EMA entirely; the target net is the only
EMA copy and ``self.model`` is what gets stored and rolled out.
"""

import copy
import logging
import math
import os
from collections import deque
from typing import Optional

import einops
import hydra
import torch
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class ConsistencyDistillationAgent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        teacher: DictConfig,
        init_from_teacher: bool,
        optimization: DictConfig,
        trainset: DictConfig,
        valset: DictConfig,
        train_batch_size,
        val_batch_size,
        num_workers,
        device: str,
        epoch: int,
        scale_data,
        target_decay: float,
        update_target_every_n_steps: int,
        num_inference_steps: int,
        goal_window_size: int,
        window_size: int,
        teacher_ckpt: Optional[str] = None,
        pred_last_action_only: bool = False,
        goal_conditioned: bool = False,
        eval_every_n_epochs: int = 50,
        checkpoint_every_n_epochs: int = 0,
        lr_warmup_steps: int = 0,
        lr_min_ratio: float = 1.0,
        grad_clip: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            trainset=trainset,
            valset=valset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device,
            epoch=epoch,
            scale_data=scale_data,
            eval_every_n_epochs=eval_every_n_epochs,
        )

        # Mirror DiffusionAgent: stash action bounds on the model. Cast to
        # float32 explicitly because numpy ``y_bounds`` is float64 and an
        # out-of-place ``torch.clamp`` would otherwise upcast x0_hat.
        self.model.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device).float()
        self.model.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device).float()

        self.eval_model_name = "eval_best_cd.pth"
        self.last_model_name = "last_cd.pth"

        self.init_from_teacher = init_from_teacher
        self.teacher = None
        if teacher_ckpt:
            self.teacher = hydra.utils.instantiate(teacher).to(self.device)
            self.teacher.min_action = self.model.min_action
            self.teacher.max_action = self.model.max_action
            if not os.path.isfile(teacher_ckpt):
                raise FileNotFoundError(f"teacher checkpoint not found: {teacher_ckpt}")
            teacher_state = torch.load(teacher_ckpt, map_location=self.device)
            self.teacher.load_state_dict(teacher_state)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            log.info("Loaded teacher from %s", teacher_ckpt)

            if init_from_teacher:
                student_param = getattr(self.model, "student_param", "eps")
                if student_param != "eps":
                    log.warning(
                        "init_from_teacher=True ignored because student_param=%s; "
                        "teacher backbone predicts eps.",
                        student_param,
                    )
                else:
                    self.model.model.load_state_dict(self.teacher.model.state_dict())
                    log.info("Student backbone warm-started from teacher backbone.")
        else:
            log.info(
                "teacher_ckpt not set; skipping frozen teacher (eval / rollout only). "
                "Training requires agents.teacher_ckpt."
            )
            if init_from_teacher:
                log.warning("init_from_teacher=True ignored because teacher_ckpt is unset.")

        # Target network for f_theta_minus (a slow-moving EMA of self.model).
        self.target_model = copy.deepcopy(self.model)
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self.target_model.eval()
        self.target_decay = target_decay
        self.update_target_every_n_steps = update_target_every_n_steps

        self.optimizer = hydra.utils.instantiate(optimization, params=self.model.get_params())
        self.steps = 0

        self.num_inference_steps = num_inference_steps
        self.goal_window_size = goal_window_size
        self.window_size = window_size
        self.pred_last_action_only = pred_last_action_only
        self.goal_condition = goal_conditioned

        self.obs_context = deque(maxlen=self.window_size)
        self.goal_context = deque(maxlen=self.goal_window_size)
        if not self.pred_last_action_only:
            self.action_context = deque(maxlen=self.window_size - 1)
            self.que_actions = True
        else:
            self.que_actions = False

        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        # LR schedule: linear warmup over ``lr_warmup_steps`` followed by
        # cosine decay to ``lr_min_ratio`` of the base LR over the rest of
        # training. ``total_steps`` is filled in lazily once we know the
        # dataloader length.
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_min_ratio = float(lr_min_ratio)
        self._base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self._total_steps = 0
        self.grad_clip = float(grad_clip)

    # ------------------------------------------------------------------
    # target-network update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_target(self) -> None:
        decay = self.target_decay
        for tp, sp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)
        for tb, sb in zip(self.target_model.buffers(), self.model.buffers()):
            tb.data.copy_(sb.data)

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------
    def _set_lr(self, step: int) -> float:
        if self._total_steps <= 0:
            return self._base_lrs[0]
        if self.lr_warmup_steps > 0 and step < self.lr_warmup_steps:
            scale = (step + 1) / max(1, self.lr_warmup_steps)
        elif self.lr_min_ratio < 1.0:
            progress = (step - self.lr_warmup_steps) / max(1, self._total_steps - self.lr_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.lr_min_ratio + (1.0 - self.lr_min_ratio) * cosine
        else:
            scale = 1.0
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            pg["lr"] = base_lr * scale
        return self._base_lrs[0] * scale

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def train_agent(self) -> None:
        best_test_metric = float("inf")

        # Establish total step count for the LR schedule.
        try:
            steps_per_epoch = len(self.train_dataloader)
        except TypeError:
            steps_per_epoch = 1
        self._total_steps = max(1, steps_per_epoch * self.epoch)

        for num_epoch in tqdm(range(self.epoch)):
            if not (num_epoch + 1) % self.eval_every_n_epochs:
                test_losses = []
                for data in self.test_dataloader:
                    if self.goal_condition:
                        state, action, mask, goal = data
                        metric = self.evaluate(state, action, goal)
                    else:
                        state, action, mask = data
                        metric = self.evaluate(state, action)
                    test_losses.append(metric)
                avg_metric = sum(test_losses) / len(test_losses)
                log.info("Epoch %d: mean eval distance is %s", num_epoch, avg_metric)
                wandb.log({"eval_distance": avg_metric, "epoch": num_epoch})
                if avg_metric < best_test_metric:
                    best_test_metric = avg_metric
                    self.store_model_weights(self.working_dir, sv_name=self.eval_model_name)
                    wandb.log({"best_model_epochs": num_epoch})
                    log.info("New best eval distance. Stored weights have been updated!")

            epoch_cd = []
            epoch_anchor = []
            for data in self.train_dataloader:
                if self.goal_condition:
                    state, action, mask, goal = data
                    cd_term, anchor_term = self.train_step(state, action, goal)
                else:
                    state, action, mask = data
                    cd_term, anchor_term = self.train_step(state, action)
                wandb.log({"cd_term": cd_term, "anchor_term": anchor_term})
                epoch_cd.append(cd_term)
                epoch_anchor.append(anchor_term)
            if epoch_cd and (num_epoch % self.eval_every_n_epochs == 0 or num_epoch == self.epoch - 1):
                log.info(
                    "Epoch %d: mean cd_term = %.6f, mean anchor_term = %.6f (steps=%d, lr=%.2e)",
                    num_epoch,
                    sum(epoch_cd) / len(epoch_cd),
                    sum(epoch_anchor) / max(1, len(epoch_anchor)),
                    self.steps,
                    self.optimizer.param_groups[0]["lr"],
                )

            if self.checkpoint_every_n_epochs > 0 and not (num_epoch + 1) % self.checkpoint_every_n_epochs:
                checkpoint_name = f"epoch_{num_epoch + 1}_cd.pth"
                self.store_model_weights(self.working_dir, sv_name=checkpoint_name)
                log.info("Stored epoch checkpoint %s", checkpoint_name)

        self.store_model_weights(self.working_dir, sv_name=self.last_model_name)
        log.info("Consistency distillation done!")

    def train_step(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        if self.teacher is None:
            raise RuntimeError(
                "CD training requires agents.teacher_ckpt (frozen DDPM). "
                "Eval-only runs can omit it."
            )
        self.model.train()
        self._set_lr(self.steps)

        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        if goal is not None:
            goal = self.scaler.scale_input(goal)

        cd_term, anchor_term = self.model.cd_loss_components(
            x_start=action,
            state=state,
            teacher=self.teacher,
            target_net=self.target_model,
            goal=goal,
        )
        loss = cd_term + self.model.lambda_direct * anchor_term

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_every_n_steps == 0:
            self._update_target()

        return float(cd_term.detach().item()), float(anchor_term.detach().item())

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor, action: torch.Tensor, goal: Optional[torch.Tensor] = None) -> float:
        state = self.scaler.scale_input(state)
        action = self.scaler.scale_output(action)
        if goal is not None:
            goal = self.scaler.scale_input(goal)

        self.model.eval()
        pred = self.model.sample(state, goal, num_inference_steps=self.num_inference_steps)
        return torch.mean((pred - action) ** 2).item()

    def reset(self) -> None:
        self.obs_context.clear()

    @torch.no_grad()
    def predict(self, state, goal: Optional[torch.Tensor] = None, extra_args=None) -> torch.Tensor:
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        state = self.scaler.scale_input(state)

        if goal is not None:
            goal = self.scaler.scale_input(goal)

        if self.window_size > 1:
            self.obs_context.append(state)
            input_state = torch.stack(tuple(self.obs_context), dim=1)
            if goal is not None:
                goal = einops.rearrange(goal, "b d -> 1 b d")
        else:
            input_state = state

        self.model.eval()

        if extra_args is not None and "num_inference_steps" in extra_args:
            num_inference_steps = int(extra_args["num_inference_steps"])
        else:
            num_inference_steps = self.num_inference_steps

        model_pred = self.model.sample(
            input_state, goal, num_inference_steps=num_inference_steps
        )
        if model_pred.dim() == 3 and model_pred.size(1) > 1:
            model_pred = model_pred[:, -1, :]

        model_pred = self.scaler.inverse_scale_output(model_pred)
        if model_pred.dim() == 3:
            model_pred = model_pred[0]
        return model_pred.cpu().numpy()

    # ------------------------------------------------------------------
    # checkpoint IO
    # ------------------------------------------------------------------
    def load_pretrained_model(self, weights_path: str, sv_name=None, **kwargs) -> None:
        ckpt_name = sv_name if sv_name is not None else "model_state_dict.pth"
        target_path = os.path.join(weights_path, ckpt_name)
        state = torch.load(target_path, map_location=self.device)
        self.model.load_state_dict(state)

        # Refresh target net so it tracks the loaded weights.
        self.target_model = copy.deepcopy(self.model)
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self.target_model.eval()
        log.info("Loaded pre-trained CD student weights from %s", target_path)

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        out_name = sv_name if sv_name is not None else "model_state_dict.pth"
        torch.save(self.model.state_dict(), os.path.join(store_path, out_name))
