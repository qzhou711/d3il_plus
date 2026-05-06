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
        "--plot-dir",
        default=None,
        help="Optional directory for per-checkpoint trajectory PNG plots.",
    )
    parser.add_argument(
        "--max-plot-trajectories",
        type=int,
        default=240,
        help="Maximum number of trajectories to draw per plot.",
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


def trim_trajectory(trajectory):
    valid = np.abs(trajectory).sum(axis=1) > 0
    if not valid.any():
        return trajectory[:0]
    return trajectory[:np.where(valid)[0][-1] + 1]


def trajectory_base_name(seed, epoch, checkpoint):
    epoch_name = safe_name(epoch if epoch is not None else checkpoint.stem)
    seed_name = "unknown" if seed is None else seed
    return f"seed_{seed_name}_epoch_{epoch_name}"


def save_trajectories(args, checkpoint, seed, epoch, sim):
    if args.trajectory_dir is None and args.plot_dir is None:
        return None, None

    trajectories = sim.last_robot_c_pos.numpy()
    successes = sim.last_successes.numpy().astype(bool)
    mode_encoding = sim.last_mode_encoding.numpy()
    modes = mode_ids(mode_encoding)
    base_name = trajectory_base_name(seed, epoch, checkpoint)

    npz_path = None
    if args.trajectory_dir is not None:
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
        )

    plot_path = None
    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir).resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{base_name}.png"
        plot_trajectories(
            plot_path,
            trajectories,
            successes,
            modes,
            seed,
            epoch,
            args.max_plot_trajectories,
        )

    return str(npz_path) if npz_path else None, str(plot_path) if plot_path else None


def plot_trajectories(plot_path, trajectories, successes, modes, seed, epoch, max_trajectories):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    total = len(trajectories)
    if max_trajectories > 0 and total > max_trajectories:
        rng = np.random.default_rng(0)
        success_idx = np.where(successes)[0]
        failure_idx = np.where(~successes)[0]
        n_success = min(len(success_idx), max_trajectories // 2)
        n_failure = max_trajectories - n_success
        chosen_success = rng.choice(success_idx, size=n_success, replace=False) if n_success else np.array([], dtype=int)
        chosen_failure = rng.choice(
            failure_idx,
            size=min(len(failure_idx), n_failure),
            replace=False,
        ) if n_failure else np.array([], dtype=int)
        draw_indices = np.concatenate([chosen_success, chosen_failure])
    else:
        draw_indices = np.arange(total)

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=160)
    ax.set_facecolor("#fbfbf7")

    obstacle_specs = [
        (0.5, -0.1, 0.03, "L1"),
        (0.425, 0.08, 0.025, "L2 top"),
        (0.575, 0.08, 0.025, "L2 bottom"),
        (0.35, 0.26, 0.025, "L3 top"),
        (0.5, 0.26, 0.025, "L3 mid"),
        (0.65, 0.26, 0.025, "L3 bottom"),
    ]
    for x, y, radius, label in obstacle_specs:
        ax.add_patch(Circle((x, y), radius, facecolor="#d73027", edgecolor="#7f0000", alpha=0.75, lw=1.0))
        ax.text(x, y + radius + 0.012, label, ha="center", va="bottom", fontsize=7, color="#7f0000")

    finish_y = -0.1 + 2.5 * 0.18
    ax.add_patch(Rectangle((0.15, finish_y - 0.01), 0.5, 0.02, facecolor="#1a9850", alpha=0.22, edgecolor="#1a9850"))
    ax.axhline(finish_y, color="#1a9850", lw=1.2, ls="--", alpha=0.8)
    ax.text(0.665, finish_y, "finish", va="center", fontsize=8, color="#1a9850")

    for y, label in [(-0.1, "level 1"), (0.08, "level 2"), (0.26, "level 3")]:
        ax.axhline(y, color="#bbbbbb", lw=0.8, ls=":", alpha=0.7)
        ax.text(0.12, y, label, va="center", fontsize=7, color="#777777")

    successful_modes = sorted(set(modes[successes]))
    cmap = plt.get_cmap("tab20", max(1, len(successful_modes)))
    mode_to_color = {mode: cmap(i % cmap.N) for i, mode in enumerate(successful_modes)}

    for idx in draw_indices:
        trajectory = trim_trajectory(trajectories[idx])
        if len(trajectory) < 2:
            continue

        if successes[idx]:
            color = mode_to_color.get(modes[idx], "#377eb8")
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=1.15, alpha=0.72, zorder=3)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=8, alpha=0.75, zorder=4)
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color="#9e9e9e", lw=0.7, alpha=0.18, zorder=2)

    ax.scatter([0.525], [-0.28], marker="*", s=110, color="#fdae61", edgecolor="#7f3b08", zorder=6, label="start")

    success_rate = float(successes.mean()) if len(successes) else 0.0
    if successes.any():
        unique_modes, counts = np.unique(modes[successes], return_counts=True)
        mode_probs = counts / counts.sum()
        entropy = float(-(mode_probs * (np.log(mode_probs) / np.log(24))).sum())
        mode_summary = sorted(zip(unique_modes, counts), key=lambda item: item[1], reverse=True)
    else:
        entropy = 0.0
        mode_summary = []

    title_epoch = epoch if epoch is not None else "unknown"
    title_seed = seed if seed is not None else "unknown"
    ax.set_title(
        f"Avoiding trajectories | seed={title_seed}, epoch={title_epoch}\n"
        f"success={success_rate:.3f}, entropy={entropy:.3f}, n={len(successes)}",
        fontsize=11,
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_xlim(0.18, 0.72)
    ax.set_ylim(-0.32, 0.42)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e0e0e0", lw=0.6, alpha=0.65)

    legend_items = [
        Line2D([0], [0], color="#9e9e9e", lw=1.5, alpha=0.45, label="failed rollout"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#fdae61", markeredgecolor="#7f3b08", markersize=10, label="start"),
        Line2D([0], [0], color="#1a9850", lw=1.5, ls="--", label="finish line"),
    ]
    for mode, count in mode_summary[:8]:
        legend_items.append(
            Line2D(
                [0],
                [0],
                color=mode_to_color.get(mode, "#377eb8"),
                lw=2,
                label=f"mode {mode}: {count}",
            )
        )
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_checkpoint(args, checkpoint):
    seed = infer_seed(checkpoint, Path(args.checkpoint_root).resolve())
    epoch = infer_epoch(checkpoint)
    cfg = build_cfg(args, seed)

    wandb.init(mode="disabled", project=cfg.wandb.project, entity=cfg.wandb.entity)

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_pretrained_model(str(checkpoint.parent), sv_name=checkpoint.name)

    sim = hydra.utils.instantiate(cfg.simulation)
    successes, entropy = sim.test_agent(agent)
    trajectory_path, plot_path = save_trajectories(args, checkpoint, seed, epoch, sim)

    wandb.finish()

    success_rate = torch.mean(successes).item()
    return {
        "seed": seed,
        "epoch": epoch,
        "success_rate": success_rate,
        "entropy": float(entropy),
        "checkpoint": str(checkpoint),
        "trajectory_path": trajectory_path,
        "plot_path": plot_path,
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
