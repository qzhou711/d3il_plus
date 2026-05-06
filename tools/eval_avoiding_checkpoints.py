import argparse
import json
import os
import re
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "environments" / "d3il"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Avoiding checkpoints and append success/entropy metrics to JSONL."
    )
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Directory that contains seed subdirectories or checkpoint files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSONL file to append evaluation metrics to.",
    )
    parser.add_argument("--config-name", default="avoiding_config")
    parser.add_argument("--agent", default="ddpm_transformer_agent")
    parser.add_argument("--agent-name", default="ddpm_transformer")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--n-timesteps", type=int, default=8)
    parser.add_argument("--n-trajectories", type=int, default=200)
    parser.add_argument("--n-cores", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--trajectory-dir",
        default=None,
        help="Optional directory for per-checkpoint trajectory NPZ files.",
    )
    parser.add_argument(
        "--pattern",
        default="epoch_*_ddpm.pth",
        help="Checkpoint filename glob relative to checkpoint-root.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be passed multiple times.",
    )
    return parser.parse_args()


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def infer_epoch(path):
    match = re.search(r"epoch_(\d+)_ddpm\.pth$", path.name)
    if match:
        return int(match.group(1))
    if path.name == "eval_best_ddpm.pth":
        return "eval_best"
    if path.name == "last_ddpm.pth":
        return "last"
    return None


def infer_seed(path, checkpoint_root):
    try:
        rel_parts = path.relative_to(checkpoint_root).parts
    except ValueError:
        rel_parts = path.parts

    for part in rel_parts:
        if part.isdigit():
            return int(part)
        match = re.search(r"(?:^|,)seed=(\d+)(?:,|$)", part)
        if match:
            return int(match.group(1))
    return None


def build_cfg(args, seed):
    overrides = [
        f"agents={args.agent}",
        f"agent_name={args.agent_name}",
        f"window_size={args.window_size}",
        "simulation.render=" + str(args.render),
        f"simulation.n_cores={args.n_cores}",
        f"simulation.n_trajectories={args.n_trajectories}",
        f"agents.model.n_timesteps={args.n_timesteps}",
    ]
    if seed is not None:
        overrides.append(f"seed={seed}")
    overrides.extend(args.extra_override)

    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), job_name="eval_avoiding_checkpoints"):
        return compose(config_name=args.config_name, overrides=overrides)


def mode_ids(mode_encoding):
    powers = 1 << np.arange(mode_encoding.shape[-1])
    return mode_encoding.astype(np.int64).dot(powers)


def trajectory_base_name(seed, epoch, checkpoint):
    epoch_name = safe_name(epoch if epoch is not None else checkpoint.stem)
    seed_name = "unknown" if seed is None else seed
    return f"seed_{seed_name}_epoch_{epoch_name}"


def save_trajectories(args, checkpoint, seed, epoch, sim):
    if args.trajectory_dir is None:
        return None

    trajectories = sim.last_robot_c_pos.numpy()
    successes = sim.last_successes.numpy().astype(bool)
    mode_encoding = sim.last_mode_encoding.numpy()
    modes = mode_ids(mode_encoding)
    base_name = trajectory_base_name(seed, epoch, checkpoint)

    trajectory_dir = Path(args.trajectory_dir).resolve()
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    npz_path = trajectory_dir / f"{base_name}.npz"
    np.savez_compressed(
        npz_path,
        trajectories=trajectories,
        successes=successes,
        mode_encoding=mode_encoding,
        mode_ids=modes,
        checkpoint=str(checkpoint),
        seed=-1 if seed is None else seed,
        epoch=-1 if not isinstance(epoch, int) else epoch,
        epoch_label=str(epoch if epoch is not None else checkpoint.stem),
    )
    return str(npz_path)


def evaluate_checkpoint(args, checkpoint):
    seed = infer_seed(checkpoint, Path(args.checkpoint_root).resolve())
    epoch = infer_epoch(checkpoint)
    cfg = build_cfg(args, seed)

    wandb.init(mode="disabled", project=cfg.wandb.project, entity=cfg.wandb.entity)

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(str(checkpoint.parent), sv_name=checkpoint.name)

    sim = hydra.utils.instantiate(cfg.simulation)
    successes, entropy = sim.test_agent(agent)
    trajectory_path = save_trajectories(args, checkpoint, seed, epoch, sim)

    wandb.finish()

    success_rate = torch.mean(successes).item()
    return {
        "seed": seed,
        "epoch": epoch,
        "success_rate": success_rate,
        "entropy": float(entropy),
        "checkpoint": str(checkpoint),
        "trajectory_path": trajectory_path,
    }


def main():
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers), replace=True)

    checkpoints = sorted(checkpoint_root.rglob(args.pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.pattern} under {checkpoint_root}")

    with output_path.open("a", encoding="utf-8") as output_file:
        for checkpoint in checkpoints:
            result = evaluate_checkpoint(args, checkpoint)
            output_file.write(json.dumps(result, sort_keys=True) + "\n")
            output_file.flush()
            print(result)


if __name__ == "__main__":
    main()
