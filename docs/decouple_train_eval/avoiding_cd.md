# Avoiding 任务上的 Consistency Distillation

本文档记录如何在已有 DDPM Transformer 教师之上做 LCM 风格的 consistency
distillation（CD），并复用现有 `eval_avoiding_checkpoints.py` 流程做对照评测。

教师 checkpoint：

```text
logs/avoiding/sweeps/ddpm_transformer/2026-05-05/19-52-56/0/eval_best_ddpm.pth
```

教师配置：`configs/agents/ddpm_transformer_agent.yaml`，cosine 调度，
`n_timesteps=8`，`predict_epsilon=true`。

## 新增模块概览

| 文件 | 作用 |
|---|---|
| `agents/models/diffusion/consistency.py` | `ConsistencyDiffusion`：student 主干 + 参数化 `f_theta` + `cd_loss` + 多步 CM 采样 |
| `agents/cd_agent.py` | `ConsistencyDistillationAgent`：装载冻结教师、维护 student EMA 和 target EMA、暴露与 `DiffusionAgent` 相同的 `predict()` 接口 |
| `configs/agents/cd_transformer_agent.yaml` | hydra 配置，需要在命令行覆写 `agents.teacher_ckpt` |
| `tools/eval_avoiding_checkpoints.py` | 扩展支持 `--num-inference-steps`、`--epoch-regex`，能直接评测 CD checkpoint |
| `tools/_smoke_cd.py` | 不依赖真实 dataset 的烟雾测试，方便快速回归 |

## 数学概览

学生预测（默认 `student_param=eps`）：

```text
eps_pred = backbone(x, t, s, g)
x0_hat   = (x - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
f_theta  = c_skip(t) * x + c_out(t) * x0_hat
```

`student_param=x0` 时 `backbone` 直接输出 `x0_hat`，跳过解析转换。

边界缩放 `boundary_scaling` 默认是 Karras / EDM 风格（CM 论文里的标准形式）：

```text
sigma(t)   = sqrt((1 - alpha_bar_t) / alpha_bar_t)
sigma_min  = sigma(t = 0)
delta(t)   = sigma(t) - sigma_min
c_skip(t)  = sigma_data^2 / (delta(t)^2 + sigma_data^2)
c_out(t)   = sigma_data * delta(t) / sqrt(sigma(t)^2 + sigma_data^2)
```

在 cosine + N=8 + sigma_data=1 下的实测系数：

```text
t   alpha_bar    sigma     c_skip   c_out
0   0.958        0.21      1.000    0.000   <- 严格 f_theta(x,0)=x
1   0.847        0.42      0.956    0.198
3   0.494        1.01      0.608    0.564
5   0.144        2.44      0.168    0.845
7   0.000        163.4     0.000    0.999   <- f_theta = x_0_hat
```

替代选项：

- `boundary_scaling: sqrt_alpha_bar` 用 `c_skip = sqrt(alpha_bar_t)`，
  `c_out = sqrt(1 - alpha_bar_t)`。在 cosine + 小 N 时高 t 处会让噪声项
  系数偏大，初版默认；保留主要为了实验对比。
- `boundary_scaling: identity` 直接 `f_theta = x_0_hat`，不做 skip。
  适合 `student_param=x0` 从零训练。

CD 训练步（**hybrid loss**，由两项构成）：

```text
# ---------- (1) consistency 项 ----------
n      ~ Uniform{0, ..., N - 2}     (N = teacher.n_timesteps = 8)
t_curr = n
t_next = n + 1
x_next = sqrt(alpha_bar_{t_next}) * x0 + sqrt(1 - alpha_bar_{t_next}) * eps
                              # 真实 (s, a) 来自 Avoiding_Dataset

# 教师做一步确定性 DDIM (frozen)
eps_t  = teacher_eps(x_next, t_next, s, g)
x0_t   = (x_next - sqrt(1 - alpha_bar_{t_next}) * eps_t) / sqrt(alpha_bar_{t_next})
x_curr = sqrt(alpha_bar_{t_curr}) * x0_t + sqrt(1 - alpha_bar_{t_curr}) * eps_t

pred   = f_theta(x_next, t_next, s, g)
target = f_theta_minus(x_curr, t_curr, s, g)    # target net 用 EMA decay = 0.99
L_cd   = pseudo_huber(pred, target)             # 默认；亦可切换 l2

# ---------- (2) direct anchoring 项（lambda_direct > 0 时启用） ----------
x_top      = randn_like(x0)                     # 与 1-step 推理输入分布一致
pred_top   = f_theta(x_top, t = N - 1, s, g)
L_anchor   = pseudo_huber(pred_top, x0)

# ---------- 总损失 ----------
L = L_cd + lambda_direct * L_anchor
```

**为什么需要 anchor 项**：在 cosine + N=8 下 `alpha_bar[N-1] ≈ 4e-5`，
`x_T` 几乎与 `x_0` 独立，1-step 推理只能依赖 `state`。纯 CD loss 有
退化解（`pred = target = constant`，loss ≈ 0），实测会卡在 `cd_loss ≈ 0.13`
不下降，eval distance 也只能短暂改善后又漂回 `~0.84`。anchor 项把 1-step
预测对齐到真实 action 分布，能把学生从这个退化解里拽出来。

## 推理（学生）

`ConsistencyDistillationAgent.predict` 调 `model.sample(state, goal,
num_inference_steps=K)`：

- `K=1`：单步 `f_theta(x_T, T)`，clamp 到 action bounds。
- `K>1`：先在 `T-1` 处一次 `f_theta`，再在等距子序列 `[T-1, ..., 1]`
  上重新加噪 + `f_theta` 交替采样。

子序列由 `linspace(T-1, 1, K)` 取整去重得到，例如：

```text
K = 1 -> [7]
K = 2 -> [7, 1]
K = 4 -> [7, 5, 3, 1]
```

## 训练命令

从仓库根目录运行：

```bash
cd /home/hulab/projects/world_action_model/d3il_plus
export PYTHONPATH=$PWD/environments/d3il:$PYTHONPATH

TEACHER_CKPT=$PWD/logs/avoiding/sweeps/ddpm_transformer/2026-05-05/19-52-56/0/eval_best_ddpm.pth

python run.py --config-name=avoiding_config \
  --multirun seed=0 \
  agents=cd_transformer_agent \
  agent_name=cd_transformer \
  window_size=5 \
  group=avoiding_cd_transformer \
  simulation.render=False \
  simulation.n_cores=10 \
  simulation.n_trajectories=200 \
  agents.teacher_ckpt=${TEACHER_CKPT} \
  agents.num_inference_steps=1 \
  epoch=300 \
  eval_every_n_epochs=20 \
  agents.checkpoint_every_n_epochs=20 \
  hydra.sweep.subdir='${seed}'
```

输出目录约定（与 DDPM sweep 一致）：

```text
logs/avoiding/sweeps/cd_transformer/<date>/<time>/0/
  eval_best_cd.pth
  last_cd.pth
  epoch_20_cd.pth
  ...
  non_ema_cd_state_dict.pth
  run.log
```

### 关键开关

| 字段 | 默认 | 说明 |
|---|---|---|
| `agents.model.student_param` | `eps` | `eps`（warm-start 友好）/ `x0`（直接预测 action）|
| `agents.model.loss_type` | `pseudo_huber` | `pseudo_huber` / `l2` |
| `agents.model.huber_c` | `7.6e-4` | pseudo-Huber 常数 |
| `agents.model.boundary_scaling` | `edm` | `edm`（推荐）/ `sqrt_alpha_bar` / `identity` |
| `agents.model.sigma_data` | `1.0` | scale 后 action 的 std，决定 EDM 系数 |
| `agents.model.lambda_direct` | `0.5` | `>0` 启用 1-step 直接锚定项；纯 CD 设 `0` |
| `agents.init_from_teacher` | `true` | `student_param=x0` 时自动忽略并打 warning |
| `agents.target_decay` | `0.99` | f_theta_minus 的 EMA decay |
| `agents.decay` | `0.995` | student inference EMA decay |
| `agents.num_inference_steps` | `1` | 默认采样步数；`predict()` 也读取 `extra_args["num_inference_steps"]` |

切换到 L2 + 直接 x0 预测：

```bash
agents.model.loss_type=l2 agents.model.student_param=x0 agents.init_from_teacher=false
```

## 评测命令（与教师对照）

`eval_avoiding_checkpoints.py` 现在同时支持 DDPM 教师和 CD 学生。CD 学生
需要传 `--num-inference-steps`，否则学生默认按 yaml 里的 1 步采样。

评测 CD **不需要** `agents.teacher_ckpt`（推理只用学生权重；配置里默认为
`null`）。若 Hydra 仍报 ``Missing mandatory value: agents.teacher_ckpt``，请
把仓库更新到当前版本。训练 CD 时仍必须在命令行指定教师 checkpoint。

```bash
SWEEP=$PWD/logs/avoiding/sweeps/cd_transformer/<date>/<time>

for K in 1 2 4; do
  python tools/eval_avoiding_checkpoints.py \
    --checkpoint-root ${SWEEP} \
    --pattern eval_best_cd.pth \
    --agent cd_transformer_agent \
    --agent-name cd_transformer \
    --num-inference-steps ${K} \
    --output results/avoiding_cd_eval_best_K${K}.jsonl \
    --n-trajectories 480 --n-cores 10 \
    --trajectory-dir results/avoiding_cd_trajectories_npz/K${K}
done
```

教师 baseline（直接复用旧命令，原行为不变）：

```bash
python tools/eval_avoiding_checkpoints.py \
  --checkpoint-root $PWD/logs/avoiding/sweeps/ddpm_transformer/2026-05-05/19-52-56 \
  --pattern eval_best_ddpm.pth \
  --output results/avoiding_ddpm_eval_best.jsonl \
  --n-trajectories 480 --n-cores 10 \
  --trajectory-dir results/avoiding_trajectories_npz
```

## JSONL 输出新增字段

CD 评测多记两列，便于后续按推理步数聚合：

```json
{"agent": "cd_transformer_agent", "num_inference_steps": 2, "epoch": "eval_best", "seed": 0,
 "success_rate": 0.83, "entropy": 0.61,
 "checkpoint": ".../0/eval_best_cd.pth",
 "trajectory_path": ".../K2/seed_0_epoch_eval_best.npz"}
```

DDPM 评测仍保持向后兼容（多了一个固定的 `"agent"` 字段）。

## 推荐对照实验表

跑完上面命令后，把结果填进下表（其中前一行是已经跑出来的教师指标）：

| 模型 | 推理步数 | success_rate | entropy | 单 step 推理时延 |
|---|---|---|---|---|
| DDPM teacher (`eval_best_ddpm.pth`) | 8 | TBD | TBD | 1× |
| CD student (`eval_best_cd.pth`) | 1 | TBD | TBD | ≈1/8× |
| CD student | 2 | TBD | TBD | ≈1/4× |
| CD student | 4 | TBD | TBD | ≈1/2× |

可视化轨迹可以直接复用 `notebooks/avoiding_trajectory_visualization.ipynb`，
把 NPZ 路径指向 `results/avoiding_cd_trajectories_npz/K{K}/...` 即可。

## 顺带修复的仓库 bug

`simulation/avoiding_sim.py` 之前把 rollout 缓存写死成 `[N, 150, 2]`。
当策略很差、episode 跑超 150 步时会直接 `RuntimeError`。现在：

- 新增 `Avoiding_Sim.max_traj_len`（默认 300），缓存按这个尺寸预分配；
- 写入时若实际长度大于缓存大小，会 truncate 并打印一行 warning。

如果想完全保留长 rollout，可以在命令行加 `simulation.max_traj_len=500`。

## 训练日志

`agents/cd_agent.py` 现在每个 eval 周期会打印一条：

```text
Epoch <num>: mean cd_loss = <float> (steps=<int>)
```

这是当个 epoch 内所有 batch 的平均 CD 损失（pseudo-Huber 或 L2）。如果
看到这一行不下降而 `eval distance` 也卡住，多半就是参数化或学习率有问题，
不是 wandb 没日志。

## 常见问题

**Q: warm-start 在 `student_param=x0` 下被忽略了，能不能强制?**

A: 不能直接 `load_state_dict`，因为输出语义变了（教师输出 ε，学生输出
x_0），整层最后一层映射意义不同。需要的话可以在 yaml 里设
`init_from_teacher=true` 同时 `student_param=eps`，然后在另一组实验里设
`init_from_teacher=false` + `student_param=x0` 做对照。

**Q: 为什么在 cd_loss 里 sample `n` 包含 0?**

A: 包含 t_curr=0 的 case 是必要的，target 在 t=0 处 `c_skip≈1, c_out≈0`，
即 target ≈ x_curr ≈ x0_teacher，自然把学生拉向真实数据流形，等价于
boundary loss。

**Q: 如何只做学生 EMA 不做 target EMA?**

A: 把 yaml 里 `agents.target_decay` 调成 1.0 会让 target 永不更新（始终是
warm-start 时的快照），通常不推荐；调小（如 0.95）会让 target 跟得更紧。

**Q: 已经跑了几百个 epoch 的 DDPM checkpoint 都能当教师吗?**

A: 可以。`agents.teacher_ckpt` 接受任意 `epoch_*_ddpm.pth` /
`eval_best_ddpm.pth` / `last_ddpm.pth`。不同教师强度下的学生也是值得做
的对照实验。

## 烟雾测试

不需要真实 dataset，只在合成 batch 上跑一次 train_step 和 K=1/2/4 推理：

```bash
python tools/_smoke_cd.py
```

通过表现：

```text
[default] CD train_step loss = ...
[default] predict K=1 / K=2 / K=4 shapes
[x0+l2] same outputs
==> save / load round-trip
max |loaded - on_disk| = 0.00e+00
ALL smoke tests OK.
```

如果 smoke test 报错先不要去跑真实训练。
