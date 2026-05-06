# Avoiding DDPM: Decoupled Training and Evaluation

本文档总结当前新增的训练与评测流程，用于在 `Avoiding` 任务上研究 DDPM / DDPM Transformer 随训练 epoch 变化的 `success_rate` 和 `entropy`，从而观察是否存在 mode collapse。

## 背景

原始流程是：

1. `run.py` 训练 agent。
2. 训练中只根据 validation loss 保存 `eval_best_ddpm.pth`。
3. 训练结束后加载 `eval_best_ddpm.pth`。
4. 调用 `simulation.avoiding_sim.Avoiding_Sim.test_agent()` 做一次 rollout eval。
5. `Successrate` 和 `entropy` 只打印到终端，不会自动保存到文件。

这个流程不方便比较不同 epoch 的 rollout 指标。因此现在采用解耦方案：

1. 训练阶段按固定 epoch 间隔保存 checkpoint。
2. 训练结束后单独运行 eval 脚本，遍历 checkpoint。
3. eval 脚本把每个 seed、每个 epoch 的指标追加写入一个 JSONL 文件。

## 新增改动

### `agents/ddpm_agent.py`

`DiffusionAgent` 新增可选参数：

```bash
checkpoint_every_n_epochs
```

默认值是 `0`，表示不额外保存 epoch checkpoint，不影响原来的训练行为。

训练时如果设置：

```bash
+agents.checkpoint_every_n_epochs=20
```

则会在每 20 个 epoch 后额外保存：

```text
epoch_20_ddpm.pth
epoch_40_ddpm.pth
epoch_60_ddpm.pth
epoch_80_ddpm.pth
epoch_100_ddpm.pth
```

原有文件仍然保留：

```text
eval_best_ddpm.pth
last_ddpm.pth
non_ema_model_state_dict.pth
```

### `tools/eval_avoiding_checkpoints.py`

新增独立评测脚本，用于：

- 遍历指定目录下的 checkpoint；
- 按 checkpoint 所在目录推断 seed；
- 按文件名推断 epoch；
- 加载 agent；
- 调用 `Avoiding_Sim.test_agent()`；
- 将 `success_rate` 和 `entropy` 写入 JSONL；
- 可选保存每个 checkpoint 的 rollout 轨迹 `.npz`；
- 可选绘制每个 checkpoint 的轨迹图 `.png`。

### `simulation/avoiding_sim.py`

`Avoiding_Sim.test_agent()` 现在会在 eval 后缓存最近一次 rollout 的原始结果：

```text
last_robot_c_pos
last_mode_encoding
last_successes
```

这样独立 eval 脚本可以在不改变原返回值的情况下，额外保存轨迹和绘图。

## 推荐训练命令

从项目根目录运行：

```bash
cd /home/qiang/projects/world_action_models/d3il
export PYTHONPATH=$PWD/environments/d3il:$PYTHONPATH

python run.py --config-name=avoiding_config \
  --multirun seed=0,1,2 \
  agents=ddpm_transformer_agent \
  agent_name=ddpm_transformer \
  window_size=5 \
  group=avoiding_ddpm_transformer_epoch_curve \
  simulation.render=False \
  simulation.n_cores=10 \
  simulation.n_trajectories=200 \
  agents.model.n_timesteps=8 \
  epoch=100 \
  eval_every_n_epochs=20 \
  +agents.checkpoint_every_n_epochs=20 \
  hydra.sweep.subdir='${seed}'
```

说明：

- `seed=0,1,2`：先跑 3 个 seed 做试验。
- `epoch=100`：训练 100 个 epoch。
- `+agents.checkpoint_every_n_epochs=20`：保存每 20 epoch 的 checkpoint。
- `hydra.sweep.subdir='${seed}'`：让每个 seed 的输出目录更清晰。
- `simulation.n_trajectories=200`：训练结束后的原始 eval 用 200 条 rollout。正式实验可改成 `480`。

输出目录通常类似：

```text
logs/avoiding/sweeps/ddpm_transformer/<date>/<time>/0/
logs/avoiding/sweeps/ddpm_transformer/<date>/<time>/1/
logs/avoiding/sweeps/ddpm_transformer/<date>/<time>/2/
```

每个 seed 目录下会包含该 seed 的 checkpoint。

## 推荐 Eval 命令

训练完成后，找到本次 sweep 的根目录，例如：

```text
logs/avoiding/sweeps/ddpm_transformer/2026-05-05/16-xx-xx
```

然后运行：

```bash
cd /home/qiang/projects/world_action_models/d3il
export PYTHONPATH=$PWD/environments/d3il:$PYTHONPATH

python tools/eval_avoiding_checkpoints.py \
  --checkpoint-root /home/qiang/projects/world_action_models/d3il/logs/avoiding/sweeps/ddpm_transformer/2026-05-05/17-25-31 \
  --output results/avoiding_ddpm_transformer_epoch_curve.jsonl \
  --n-trajectories 480 \
  --n-cores 10
```

如果想同时保存轨迹数据和轨迹图：

```bash
python tools/eval_avoiding_checkpoints.py \
  --checkpoint-root /home/qiang/projects/world_action_models/d3il/logs/avoiding/sweeps/ddpm_transformer/2026-05-05/17-25-31 \
  --output results/avoiding_ddpm_transformer_epoch_curve.jsonl \
  --n-trajectories 480 \
  --n-cores 10 \
  --trajectory-dir results/avoiding_trajectories_npz \
  --plot-dir results/avoiding_trajectory_plots \
  --max-plot-trajectories 240
```

如果想正式评测，可以把 rollout 数量改成：

```bash
--n-trajectories 480
```

默认会评测：

```text
epoch_*_ddpm.pth
```

如需评测最佳 checkpoint，可以指定：

```bash
--pattern eval_best_ddpm.pth
```

如需评测最后 checkpoint，可以指定：

```bash
--pattern last_ddpm.pth
```

轨迹相关参数：

```text
--trajectory-dir
  保存每个 checkpoint 的轨迹、成功标记、mode encoding 和 mode id，格式为 .npz。

--plot-dir
  保存每个 checkpoint 的轨迹可视化，格式为 .png。

--max-plot-trajectories
  每张图最多绘制多少条轨迹，默认 240。轨迹太多时会固定随机种子抽样，避免图过于拥挤。
```

## JSONL 输出格式

输出文件每行是一条 JSON，例如：

```json
{"checkpoint": ".../0/epoch_20_ddpm.pth", "entropy": 0.31, "epoch": 20, "plot_path": ".../seed_0_epoch_20.png", "seed": 0, "success_rate": 0.12, "trajectory_path": ".../seed_0_epoch_20.npz"}
{"checkpoint": ".../0/epoch_40_ddpm.pth", "entropy": 0.55, "epoch": 40, "plot_path": ".../seed_0_epoch_40.png", "seed": 0, "success_rate": 0.24, "trajectory_path": ".../seed_0_epoch_40.npz"}
{"checkpoint": ".../1/epoch_20_ddpm.pth", "entropy": 0.28, "epoch": 20, "plot_path": ".../seed_1_epoch_20.png", "seed": 1, "success_rate": 0.10, "trajectory_path": ".../seed_1_epoch_20.npz"}
```

后续可以用这个文件画曲线：

- x 轴：`epoch`
- y 轴 1：`success_rate`
- y 轴 2：`entropy`
- group：`seed`

## 轨迹文件和可视化

如果指定 `--trajectory-dir`，每个 checkpoint 会保存一个 `.npz` 文件，包含：

```text
trajectories    shape: [n_trajectories, max_steps, 2]
successes       shape: [n_trajectories]
mode_encoding   shape: [n_trajectories, 9]
mode_ids        shape: [n_trajectories]
checkpoint      checkpoint 路径
seed            seed
epoch           epoch
```

其中 `trajectories` 是机器人末端在 xy 平面上的 rollout 轨迹。无效 padding 点为 0，绘图时会自动裁掉。

如果指定 `--plot-dir`，每个 checkpoint 会保存一张 `.png` 轨迹图。图中元素含义：

```text
彩色轨迹：成功 rollout，不同颜色表示不同 mode。
灰色半透明轨迹：失败 rollout。
红色圆：障碍物。
虚线水平线：三层障碍通道参考线。
绿色线/区域：finish line。
黄色星号：起点。
图例：主要成功 mode 及其轨迹数量。
标题：seed、epoch、success_rate、entropy 和 rollout 数量。
```

这些图适合直观看 mode collapse：

```text
如果成功轨迹几乎全是同一种颜色、走同一条通道，即使 success_rate 较高，也可能说明策略退化到单一行为。
如果成功轨迹颜色多、覆盖多条通道，说明多模态保持较好。
如果灰色失败轨迹很多，则应先关注策略是否学会任务，而不是只看 mode collapse。
```

## 指标解释

`success_rate` 表示 rollout 成功率。

在 `Avoiding` 中，一条 rollout 成功代表机器人末端绕过障碍并到达目标区域。如果撞到障碍，或没有到达目标，则失败。

`entropy` 表示成功轨迹中的 mode 多样性。

`Avoiding` 有 3 层障碍通道：

```text
第 1 层：2 种路径选择
第 2 层：3 种路径选择
第 3 层：4 种路径选择
```

理论组合最多有：

```text
2 x 3 x 4 = 24 modes
```

评测代码只对成功轨迹统计 mode 分布，然后计算归一化熵：

```text
entropy 越高：成功轨迹覆盖更多行为模式
entropy 越低：成功轨迹越集中，可能存在 mode collapse
```

判断时不要只看 entropy：

```text
高 success_rate + 高 entropy：多模态保持较好
高 success_rate + 低 entropy：可能退化到单一成功行为
低 success_rate + 低 entropy：模型可能还没学会，不一定只是 mode collapse
低 success_rate + 高 entropy：行为多样但不稳定
```

## 注意事项

当前已经启动的训练进程不会自动拥有 `checkpoint_every_n_epochs` 功能。只有在启动命令中显式加入：

```bash
+agents.checkpoint_every_n_epochs=20
```

并重新开始训练，才会生成 `epoch_*_ddpm.pth`。

`eval_avoiding_checkpoints.py` 会追加写入 `--output` 指定的 JSONL 文件。如果不想混入旧结果，请先换一个新的输出文件名，或手动删除旧文件。

`--trajectory-dir` 和 `--plot-dir` 也会按 checkpoint 生成文件。如果重复评测同一个 checkpoint，同名 `.npz` / `.png` 会被覆盖，但 JSONL 会继续追加新行。

多进程评测建议保持：

```bash
simulation.render=False
--n-cores 10
```

如果要可视化单条 rollout，应使用：

```bash
--render --n-cores 1 --n-trajectories 1
```

