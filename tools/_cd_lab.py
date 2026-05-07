"""Single-shot autonomous experiment runner for CD on Avoiding.

For each (gpu_id, training_args, eval_args) entry, this script:
  1. Launches a CD training run with hydra overrides on a dedicated GPU.
  2. Waits for it to finish.
  3. Runs rollout eval on eval_best_cd.pth and last_cd.pth at K=1/2/4.
  4. Appends a one-line JSON summary per (run, K, ckpt) into ``--master-jsonl``.

Designed to be the main coordinator while iterating overnight.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd, env=None, capture=True):
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, env=env, capture_output=capture, text=True)


def launch_train(name, sweep_dir, overrides, gpu, teacher_ckpt, epoch, eval_every, ckpt_every, log_path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO}/environments/d3il:{env.get('PYTHONPATH','')}"

    cmd = [
        PYTHON, str(REPO / "run.py"),
        "--config-name=avoiding_config", "--multirun",
        "seed=0",
        "agents=cd_transformer_agent",
        "agent_name=cd_transformer",
        "window_size=5",
        f"group={name}",
        "simulation.render=False",
        "simulation.n_cores=1",
        "simulation.n_trajectories=1",
        f"agents.teacher_ckpt={teacher_ckpt}",
        "agents.num_inference_steps=1",
        f"epoch={epoch}",
        f"eval_every_n_epochs={eval_every}",
        f"agents.checkpoint_every_n_epochs={ckpt_every}",
        f"hydra.sweep.dir={sweep_dir}",
        "hydra.sweep.subdir=${seed}",
    ] + overrides

    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def eval_ckpts(name, sweep_dir, gpu, master_jsonl, n_traj=100, n_cores=10, ks=(1, 2, 4), patterns=("eval_best_cd.pth", "last_cd.pth")):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{REPO}/environments/d3il:{env.get('PYTHONPATH','')}"

    tmp_jsonl = Path(sweep_dir) / "rollout_eval.jsonl"
    if tmp_jsonl.exists():
        tmp_jsonl.unlink()

    for K in ks:
        for pattern in patterns:
            cmd = [
                PYTHON, str(REPO / "tools" / "eval_avoiding_checkpoints.py"),
                "--checkpoint-root", str(sweep_dir),
                "--pattern", pattern,
                "--agent", "cd_transformer_agent",
                "--agent-name", "cd_transformer",
                "--num-inference-steps", str(K),
                "--output", str(tmp_jsonl),
                "--n-trajectories", str(n_traj),
                "--n-cores", str(n_cores),
            ]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{name}] eval K={K} {pattern} FAILED:\n{r.stderr[-500:]}", flush=True)
                continue

    # Tag with experiment name and append to master jsonl.
    if not tmp_jsonl.exists():
        return
    with master_jsonl.open("a") as fout, tmp_jsonl.open() as fin:
        for line in fin:
            row = json.loads(line)
            row["experiment"] = name
            fout.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[{name}] appended {sum(1 for _ in tmp_jsonl.read_text().splitlines() if _.strip())} eval rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-jsonl", default=str(REPO / "results" / "cd_lab.jsonl"))
    ap.add_argument("--teacher-ckpt", default=str(REPO / "logs/avoiding/sweeps/ddpm_transformer/2026-05-05/19-52-56/0/eval_best_ddpm.pth"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--n-traj", type=int, default=100)
    ap.add_argument("--n-cores", type=int, default=10)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    master = Path(args.master_jsonl)
    master.parent.mkdir(parents=True, exist_ok=True)

    sweep_dir = REPO / "logs/avoiding/sweeps_v2" / args.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / "logs/avoiding/sweeps_v2" / f"{args.name}.log"

    print(f"=== {args.name} on GPU{args.gpu} ===", flush=True)
    proc, log_f = launch_train(
        name=args.name, sweep_dir=str(sweep_dir),
        overrides=args.override, gpu=args.gpu,
        teacher_ckpt=args.teacher_ckpt,
        epoch=args.epoch, eval_every=args.eval_every, ckpt_every=args.ckpt_every,
        log_path=str(log_path),
    )
    proc.wait()
    log_f.close()
    rc = proc.returncode
    print(f"[{args.name}] training exit={rc}", flush=True)
    if rc != 0:
        return

    eval_ckpts(
        name=args.name, sweep_dir=str(sweep_dir),
        gpu=args.gpu, master_jsonl=master,
        n_traj=args.n_traj, n_cores=args.n_cores,
    )
    print(f"[{args.name}] done.", flush=True)


if __name__ == "__main__":
    main()
