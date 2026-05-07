"""Smoke test for the consistency-distillation stack.

Instantiates ConsistencyDistillationAgent through Hydra exactly the way
``run.py`` does, then runs:

1. one CD train_step on a tiny synthetic batch (no real dataset);
2. one inference call via predict() at K=1 and K=2.

This avoids touching the real Avoiding dataset / simulation while exercising
all newly added code paths.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "environments" / "d3il"))


def _build_agent(overrides):
    import hydra

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), job_name="smoke_cd"):
        cfg = compose(config_name="avoiding_config", overrides=overrides)

    if wandb.run is None:
        wandb.init(mode="disabled", project=cfg.wandb.project, entity=cfg.wandb.entity)
    agent = hydra.utils.instantiate(cfg.agents)
    return agent, cfg


def _exercise(agent, cfg, label):
    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    window_size = cfg.window_size

    state = torch.randn(4, window_size, obs_dim, device=cfg.device)
    action = 0.1 * torch.randn(4, window_size, action_dim, device=cfg.device)

    cd_term, anchor_term = agent.train_step(state, action)
    print(f"[{label}] CD train_step cd_term={cd_term:.6f} anchor_term={anchor_term:.6f}")
    assert np.isfinite(cd_term) and np.isfinite(anchor_term), "Loss is not finite"

    agent.reset()
    obs_np = np.random.randn(obs_dim).astype(np.float32)
    pred_k1 = agent.predict(obs_np)
    print(f"[{label}] predict K=1 shape: {pred_k1.shape} mag={np.abs(pred_k1).mean():.4f}")

    agent.reset()
    pred_k2 = agent.predict(obs_np, extra_args={"num_inference_steps": 2})
    print(f"[{label}] predict K=2 shape: {pred_k2.shape} mag={np.abs(pred_k2).mean():.4f}")
    assert pred_k1.shape == pred_k2.shape

    agent.reset()
    pred_k4 = agent.predict(obs_np, extra_args={"num_inference_steps": 4})
    print(f"[{label}] predict K=4 shape: {pred_k4.shape} mag={np.abs(pred_k4).mean():.4f}")


def main() -> None:
    OmegaConf.register_new_resolver("add", lambda *xs: sum(xs), replace=True)

    teacher_ckpt = (
        REPO_ROOT
        / "logs/avoiding/sweeps/ddpm_transformer/2026-05-05/19-52-56/0/eval_best_ddpm.pth"
    )
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"missing teacher ckpt: {teacher_ckpt}")

    base_overrides = [
        "agents=cd_transformer_agent",
        "agent_name=cd_transformer",
        "window_size=5",
        "epoch=1",
        "eval_every_n_epochs=1",
        f"agents.teacher_ckpt={teacher_ckpt}",
        "agents.num_inference_steps=1",
        "train_batch_size=8",
        "val_batch_size=8",
        "num_workers=2",
    ]

    print("==> default (eps + pseudo_huber + warm-start + hybrid lambda=0.5)")
    agent, cfg = _build_agent(list(base_overrides))
    print("Student params:", sum(p.numel() for p in agent.model.get_params()))
    print("Teacher params:", sum(p.numel() for p in agent.teacher.model.parameters()))
    print("Target net id != student id:", id(agent.target_model) != id(agent.model))
    _exercise(agent, cfg, "default")

    print("==> pure CD (lambda_direct=0)")
    agent_pure, cfg_pure = _build_agent(
        list(base_overrides) + ["agents.model.lambda_direct=0.0"]
    )
    _exercise(agent_pure, cfg_pure, "pure-cd")

    print("==> l2 loss + x0 student + warm-start ignored + heavy anchor")
    agent_x0, cfg_x0 = _build_agent(
        list(base_overrides)
        + [
            "agents.model.loss_type=l2",
            "agents.model.student_param=x0",
            "agents.model.lambda_direct=2.0",
        ]
    )
    _exercise(agent_x0, cfg_x0, "x0+l2+anchor2")

    print("==> save / load round-trip")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        agent.store_model_weights(tmp, sv_name="eval_best_cd.pth")
        on_disk = torch.load(os.path.join(tmp, "eval_best_cd.pth"), map_location="cpu")
        for p in agent.model.parameters():
            p.data.add_(torch.randn_like(p.data))
        agent.load_pretrained_model(tmp, sv_name="eval_best_cd.pth")
        loaded = agent.model.state_dict()
        max_diff = max(
            (on_disk[k].cpu() - loaded[k].cpu()).abs().max().item() for k in on_disk
        )
        print(f"max |loaded - on_disk| = {max_diff:.2e}")
        assert max_diff < 1e-6, "save/load mismatch"

    print("ALL smoke tests OK.")


if __name__ == "__main__":
    main()
