#!/bin/bash
# Usage: _run_cd_eval.sh <name> <gpu> <sweep_dir>
set -e
NAME=$1
GPU=$2
SWEEP=$3
cd /home/hulab/projects/world_action_model/d3il_plus
source /home/hulab/anaconda3/etc/profile.d/conda.sh
conda activate d3il
export PYTHONPATH=$PWD/environments/d3il:$PYTHONPATH
mkdir -p results
OUT=results/cd_eval_${NAME}.jsonl
rm -f "$OUT"
for K in 1 2 4; do
  for P in eval_best_cd.pth last_cd.pth; do
    echo "[$NAME GPU$GPU] K=$K $P"
    CUDA_VISIBLE_DEVICES=$GPU python tools/eval_avoiding_checkpoints.py \
      --checkpoint-root "$SWEEP" \
      --pattern "$P" \
      --agent cd_transformer_agent \
      --agent-name cd_transformer \
      --num-inference-steps "$K" \
      --output "$OUT" \
      --n-trajectories 100 \
      --n-cores 10 2>&1 | tail -1
  done
done
echo "[$NAME] DONE -> $OUT"
