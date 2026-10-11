# 训练 / 评估标准 r2（2026-09-29 起生效，2026-09-30 复核）

> **2026-09-30：训练已由 [`standard_r3.md`](standard_r3.md) 取代。** r2 的 SIGReg 在展平的 17 个 token 上计算，会让 S2 坍缩（point 上 2 个 epoch 已复现）。正式训练一律用 `+recipe=r3`（= r2 + 逐 token SIGReg），评估前先跑 `scripts/repr_health.py`。

> 取代此前所有训练配方和评估设定（下文称 r1）。r1 的结果只作历史参考，**不与 r2 混表**。  
> 原则：**训练对齐官方 LeWM**（`lucas-maes/le-wm`，`config/train/lewm.yaml` + `train.py`）；**评估对齐已发表协议**（规划设定取 DINO-WM，导航预算取 LeWM）；**调参只在 dev 划分上做**，test 划分只出最终数字。  
> token / pool / pos / cost 等方法相关 CLI 仍以 [`S1_S2_eval_protocol.md`](S1_S2_eval_protocol.md) §1.3–1.5 为准；时程、MPC、n、划分以本文为准。

---

## 0. 为什么要换（r1 的问题）

| 问题 | 原因 | 影响 |
|------|------|------|
| **LR 不退火** | spt 的手动优化补丁把 `accumulate_grad_batches` 改成 1，默认调度总步数按 batch 数算，高估了累积倍数 | 累积 8 时训练结束 LR 仍为峰值的约 97%，累积 4 时约 87%；最后一个 ckpt 停在高 LR 上 |
| **累积梯度未平均** | `spt.Module.training_step` 直接对原始 loss 反传 | 梯度是 N 个 micro-batch 相加；grad clip 裁的是放大 N 倍的梯度 |
| **BatchNorm / SIGReg 只看到 16 个样本** | projector 和 pred_proj 用 `BatchNorm1d`；SIGReg 在每个 micro-batch 内算统计量 | 梯度累积救不回来，官方是真 batch 128 |
| **超参被逐环境改过** | wall wd 2e-3、rope lr 3e-5、λ 0.05 | 是在上面三个 bug 下、用 n=10 评估选出来的，不可信 |
| **评估不可复现** | 评估脚本没给 torch 设种子；torch 默认每个进程随机取种子 | 同一 ckpt、同一命令重跑，CEM 采样都不同；不同方法在同一 episode 上噪声也不同 |
| **wall 评估实际跑在 CPU 上** | `eval_wall.py` 默认规划卡是 2，本机只有 0/1，脚本静默退回 CPU；point 默认 7，直接渲染失败 | CPU 与 GPU 数值不同，且慢约 30 倍 |
| **MPC 参数在测试集上挑** | wall 2/10、point 5/10 是在同一批测试 episode 上用 S1/S2 权重选的 | 偏向自己的方法；且不是任何已发表协议 |

r1 时期 8 卡机上的 CLS / C1 也是 2 卡 ×4 累积，**同样有前三个问题**。

## 1. 代码改动

训练（`train.py`、`config/train/recipe/r2.yaml`）：

- 调度器按真实优化器步数：`optimizer_steps_total()` = ⌊每卡 batch 数 / 累积数⌋ × `max_epochs`，1% 线性 warmup 后余弦到 0。不累积时与官方完全相同。
- `accumulate_grad_batches > 1` 时 loss 除以累积数（r2 不累积，只是防坑）。
- `loss.sigreg.fp32`：默认 `false`，与官方一样在 bf16 autocast 下计算。
- `wm.encoder_grad_ckpt`：编码器 gradient checkpointing，不改变结果（§2.3）。
- `trainer.deterministic: true`：只换确定性 GPU 算子，重跑逐位一致（§2.4）。

评估（`eval_{wall,point_maze,rope_1,granular}.py`、`eval_mpc_utils.py`）：

- 每个 episode 开始时 `torch.manual_seed(seed + i)`。
- `--split dev|test|all`（默认 `dev`），见 §4.2。
- 规划卡默认 0、渲染卡默认 1；请求 CUDA 却不可用时直接报错，不再静默退回 CPU。
- 读 `*_object.ckpt` 时，`inject_every_layer` / `n_lo_mode` 默认沿用 ckpt 里训练时的值；只有显式传参才覆盖。以前会被 CLI 默认值悄悄改掉，评估 S2 漏传 `--inject-every-layer` 就等于换了模型。
- rope/granular 的规划文件默认改为 `<--output 去后缀>_plans.pkl`。以前固定写 `outputs/{rope,granular}_plans.pkl`，并发评估会互相覆盖。
- wall/point MPC 记录每轮后的距离 `dist_history`，默认 `--n-taken-actions 5 --mpc-max-iters 10`。
- `scripts/mpc_budget_curve.py`：从一次 5/10 结果读 success@k，带 Wilson 区间和配对 McNemar。

## 2. 训练标准 r2

### 2.1 配方（四环境、所有方法统一）

| 项 | r2 | 官方 LeWM | r1 |
|----|----|-----------|----|
| batch | **128，不累积** | 128，不累积 | 16 × 累积 4/8 |
| 设备 | 单卡 `[0]` | 单卡（`devices: auto`） | 1 或 2 卡混用 |
| 精度 | bf16-mixed，SIGReg 也在 bf16 下 | `bf16`（同义） | bf16-mixed |
| 优化器 | AdamW，lr **5e-5**，wd **1e-3** | 同 | 各环境不同 |
| LR 调度 | 1% warmup + 余弦到 0，按步 | 同 | 实际基本恒定 |
| grad clip | 1.0（按范数） | 同 | 1.0（裁的是放大后的梯度） |
| SIGReg | λ **0.09**，knots 17，num_proj 1024 | 同 | λ 0.05 / 0.09 |
| predictor | depth 6，heads 16，mlp 2048，dim_head 64，dropout 0.1 | 同 | 同 |
| history / num_preds / embed_dim | 3 / 1 / 192 | 同 | 同 |
| max_epochs | **100**（§3） | 代码默认 100 | CLS wall 60 / 其它 100 / S2 150 |
| seed | 3072 | 3072 | 3072 |
| 编码器 grad ckpt | 开 | 关 | 关 |
| 确定性算子 | 开 | 关 | 关 |
| 存档 | 每 10 epoch 一个 object，外加 `*_last_object.ckpt` | 每 epoch | 同 r2 |

与官方的差异只有 grad ckpt 和确定性算子，两者都不改变数学。frameskip 由数据决定：wall/point 为 5，rope/granular 为 1（DINO-WM 数据集本身的设定）。`num_workers` / `prefetch_factor` 只影响速度。

每 epoch 步数（bs 128）：wall ≈376，point ≈1025，rope / granular ≈113。

**官方核对**（2026-09-29，`train.py` 共 6 次提交，83f97d7 → 8edfeb3）：

- 训练超参从首次提交起没变过；模型结构（ViT-tiny/14、ARPredictor、projector 和 pred_proj 为 2048 隐层 + BatchNorm1d）与本地 `build_lewm` 一致。
- 官方写的是 `"interval": "epoch"`，但 spt 手动优化在每次 `opt.step()` 后都调用 `scheduler.step()`，这个字段不起作用，官方实际按步退火。
- 本地 `config/train/lewm.yaml` 被改过（agibot 数据、50 epoch、4 卡、fp32）。四个环境的 yaml 都不继承它，不影响这里的训练，保持原样未动。

**本地复核**（2026-09-30）：

| 项 | 结论 |
|----|------|
| 种子 | `seed_everything(3072, workers=True)`；train/val 切分和 DataLoader 打乱共用 `Generator(3072)` |
| grad clip | spt 把 Trainer 的 1.0 转交给手动优化，日志中 `clip val. = 1.0` |
| 训练/评估数据隔离 | 按 episode 首帧像素和前 5 步动作做指纹，eval 与 train **零重叠**（wall/point 首帧、动作均零重叠；rope/granular 所有 episode 首帧相同，是数据集设定，动作零重叠） |
| 归一化统计量 | 训练和评估都用 `*_train.h5` 全量（torch 无偏 std 对 sklearn 有偏 std，差约 3e-5，可忽略） |
| `last_object` | 在 `epoch == max_epochs` 时由训练中的模型直接 pickle，BN 统计量即训练结束时的值 |

### 2.2 显存与耗时（4090）

| 方法 | batch | grad ckpt | 峰值显存 |
|------|------:|:---------:|---------:|
| S2 | 128 | 关 | 约 17.3 GiB（推算） |
| **S2** | **128** | **开** | **6.8 GiB**（确定性模式下 point 实测 7.8 GiB 整卡占用） |
| CLS | 128 | 开 | 3.0 GiB |

S2 实测（point，确定性模式，数据已在页缓存中）：约 0.29 s/batch。据此估 100 epoch：rope/granular 约 1h，wall 约 3h，point 约 9h；CLS 更快。  
两张卡都空闲时可以各跑一个训练（`CUDA_VISIBLE_DEVICES=1` 即用 1 号卡）；rope/granular 评估需要另占一张卡做 EGL 渲染。

### 2.3 gradient checkpointing 为什么不影响结果

- 反传时重算编码器前向，而不是存激活；同一个函数、同样的输入，数学上完全一样。
- 实测：开和不开时各参数梯度逐元素相同（相对差 0）。
- 编码器 dropout 为 0；即使有 dropout，`torch.utils.checkpoint` 也会保存并恢复 RNG 状态。

### 2.4 可复现性实测

同一配置连跑两次（1 epoch × 20–30 batch），逐参数比较 `last_object`：

| 设定 | 结果 |
|------|------|
| 默认（非确定性算子） | 320 个张量中 318 个不同，30 步后 `pred_loss` 0.2295 对 0.2204 |
| `trainer.deterministic=true`（S2 与 CLS） | **全部张量逐位相同** |

说明种子本身没问题，差异来自 GPU 原子加等非确定性算子。确定性模式慢约 10%（3.73 对 4.15 it/s），显存不变，r2 默认开启。Lightning 会自己设置 `CUBLAS_WORKSPACE_CONFIG`。跨卡逐位一致只对同型号、同驱动成立（本机两张 4090）。

### 2.5 训练命令

```bash
export STABLEWM_HOME=/data/czx/le-wm/first/data
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python train.py --config-name=lewm_rope +recipe=r2        # CLS
.venv/bin/python train.py --config-name=lewm_rope_c1 +recipe=r2     # C1
.venv/bin/python train.py --config-name=lewm_rope_s1 +recipe=r2     # S1
.venv/bin/python train.py --config-name=lewm_rope_s2 +recipe=r2     # S2
# 改轮数 / seed：追加 trainer.max_epochs=50 seed=1234
```

中断后续训：用同一个 `subdir=<原目录名>` 重跑（读取 `*_weights.ckpt`，优化器、调度器、epoch 一起恢复）。

## 3. 训练轮数

### 3.1 参照

- 官方 LeWM 代码默认 100 epoch；论文在自己的环境上训 10 epoch，但那些数据集大得多。
- DINO-WM 在**这四个数据集**上训 100 epoch（`conf/train.yaml`，batch 32，编码器冻结）。
- 本仓库 100 epoch 的步数：rope/granular 约 1.1 万，wall 约 3.8 万，point 约 10 万。

默认 **100**；rope/granular 和 wall 步数偏少，重点检查"是否需要更多轮"。

### 3.2 不能用一个长 run 的中间 ckpt 代替短 run

一个 200 epoch 余弦 run 在第 50 epoch 时 LR 还是峰值的 0.85，第 100 epoch 是 0.50；单独训 50 或 100 epoch 的 run 最后都退火到 0。中间 ckpt 没经过退火，结果系统性偏差且更抖，所以**每个候选轮数都要单独训练**。

### 3.3 试点：从 50、100 开始，好就翻倍（在 dev 上选）

每个环境用 CLS 和 S2：

1. 先训 E=50 和 E=100，用 **dev** 划分评估。
2. 若 100 比 50 好且超过 §4.5 的噪声阈值，再训 200；200 仍更好再训 400。
3. 取**最小的 E**，要求其 CLS 和 S2 都与最优 E 相差在噪声以内；平手选 100。
4. 同一环境内所有方法用同一个 E。

| 组 | 环境 | 起始 E | 额外 | 训练耗时（估，CLS+S2，50+100） |
|----|------|--------|------|------------------------------|
| P1 | rope, granular | 50 / 100 | E=100 加 seed 1234（测噪声） | 约 8h |
| P2 | wall | 50 / 100 | — | 约 8h |
| P3 | point | 50 / 100 | — | 约 22h |

E=100 的 run 同时就是正式基线。

## 4. 评估标准 r2

### 4.1 权重

- **一律用 `*_last_object.ckpt`**，不再用按 val `pred_loss` 选的 best（JEPA 目标是自己的 embedding，早期 loss 天然偏低，r1 的 best 常落在 ep2、ep4）。
- 所有方法同一轮数、同一配方训练。

### 4.2 dev / test 划分

`*_eval.h5` 按 episode 编号奇偶切成两半，互不重叠（已验证）：

| 划分 | episode | 用途 |
|------|---------|------|
| `--split dev`（默认） | 奇数编号 | 选轮数、测噪声底、扫 cost、筛训练改动等**一切选择** |
| `--split test` | 偶数编号 | 只出论文数字；每个方法版本只跑一次，不看结果回头改 |

每个划分的 episode 数：wall 96，point 100，rope 50，granular 50。n 取满，每个 episode 只用一次。

### 4.3 规划协议：一套规则，四个数据集共用

规划设定以 **DINO-WM 的 plan 配置**为准（四个数据集都来自 DINO-WM），导航任务的有限预算取自 **LeWM 论文**（预算 = 2 × 目标距离）。

**所有环境相同：**

- 目标：数据集轨迹 +g 处的状态，g = `goal_H` 5 × frameskip，wall/point 为 **25**，rope/granular 为 **5**。
- `--plan-protocol lewm`（默认）：规划只用当前 1 帧作上下文（`swm.PlanConfig.history_len=1`），CEM 只优化 horizon 段，返回最后一轮 elite 均值。更新规则已与 `stable_worldmodel.solver.CEMSolver` 逐行核对：第 0 个候选固定为当前均值，elite std 用无偏估计；`n_taken = horizon` 时官方的 warm start 为空。
- CEM：horizon 5，300 样本，top 30，var 1。
- 每轮执行全部 5 步（`n_taken = 5`），每轮结束检查一次，成功即停。
- 每个 episode 开始时 `torch.manual_seed(seed + i)`。

**按任务类型不同**（与 DINO-WM 自己的划分一致）：

| 任务 | 环境 | 轮数 | CEM 迭代 | 依据 |
|------|------|------|---------|------|
| 导航 | wall, point | 最多 10 轮；**主指标 success@2**（50 环境步） | 10 | DINO-WM `plan_wall` / `plan_point_maze` + LeWM 预算 50 |
| 可变形体 | rope, granular | 1 轮（开环） | 30 | DINO-WM `plan.yaml` |

- 开环就是 MPC 的第 1 轮：g=5、fs=1 时一次 5 步规划正好覆盖目标距离。
- wall/point 只跑一次 5/10，用 `scripts/mpc_budget_curve.py` 读 success@k。每个 episode 单独设种子，所以前 k 轮与单独跑 `--mpc-max-iters k` **逐 episode 相同**（已验证）。主表 success@2，同表副栏 success@10（250 步）。
- r1 的 wall 2/10 只在附录里对最终 CLS 和 S2 做规划器敏感性对照。
- DINO-WM 的 `random_state` 目标（wall/point）作为最终方法的补充表，见 §6。

### 4.4 各环境评估命令

point / rope / granular 先 `source setup_env.sh`（只设 MuJoCo、PyFleX、EGL 的环境变量，默认规划卡 0、渲染卡 1）；Python 一律用 `.venv/bin/python`。

| 环境 | CLI（dev 调参时去掉 `--split test`） | n | seed | 主指标 |
|------|------|--:|-----:|--------|
| wall | `eval_wall.py --split test --num-eval 96 --goal-offset 25 --planner mpc --n-taken-actions 5 --mpc-max-iters 10 --cem-num-samples 300 --cem-topk 30 --cem-n-steps 10` | 96 | 42 | success@2（`‖Δxy‖<4.5`），副栏 success@10 |
| point_maze | `eval_point_maze.py --split test --num-eval 100`，其余同 wall | 100 | 42 | success@2（`‖Δxy‖<0.5`），副栏 success@10 |
| rope | `eval_rope_1.py --split test --num-eval 50 --goal-offset 5 --horizon 5 --cem-num-samples 300 --cem-topk 30 --cem-n-steps 30` | 50 | 42 | Chamfer 均值 / 中位数（成功率 <1.0 放附录） |
| granular | `eval_granular.py --split test --num-eval 50 --goal-offset 5 --horizon 5 --planner openloop --cem-num-samples 300 --cem-topk 30 --cem-n-steps 30` | 50 | 42 | 同上 |

读 wall/point 曲线（第一个文件为基线，其余与它做配对 McNemar）：

```bash
.venv/bin/python scripts/mpc_budget_curve.py outputs/wall_cls_details_r2.json outputs/wall_s2_details_r2.json \
    --frameskip 5 --n-taken 5 --ks 1 2 4 10
```

注意：

- 修复前（2026-09-29 之前）的评估数字都不能和修复后的混用。
- rope 只用 `eval_rope_1.py`。旧的 `eval_rope.py` 默认 g15、CEM 128/5/16/var 0.5，**不要用**。
- `eval_granular.py` 默认 `planner=mpc`，必须显式传 `--planner openloop`。
- wall/point 同时报告起点已在成功球内（`mpc_iters=0`）的 episode 数。
- cost：CLS 用 `dense_terminal`；C1/S1/S2 用 `m_task_terminal`（eps 0.08），参数按协议文档 §1.3–1.5。
- 日志里核对一行 `[INFO] token_mode=... pos_mode=... inject_every_layer=...`，确认与训练配置一致。

### 4.5 可复现性实测

| 环境 | 重跑结果 |
|------|---------|
| wall（GPU，6 个 episode） | 每个 episode 的成功与否、终点距离、每轮距离全部相同 |
| point（GPU，4 个 episode） | 同上 |
| rope（2 个 episode） | 规划逐位相同（规划 cost 完全一致）；FleX 物理仿真本身有约 1e-4 的非确定性（Chamfer 1.986573 对 1.986356），比方法间差距小约三个数量级 |
| granular | 与 rope 同一套规划代码和 FleX 仿真 |

实测耗时（GPU）：wall 约 1 秒/episode（5/10）；rope/granular 以仿真为主，约 40 秒/episode。

### 4.6 报告格式

- 成功率：百分比 + Wilson 95% 置信区间；方法间差异用配对 McNemar（同一批 episode、同一组 CEM 噪声）。
- Chamfer：均值 ± bootstrap 95% 置信区间（10k 次重采样），另报中位数；方法间差异用配对 bootstrap。
- 结果文件名带划分和 `_r2`，例如 `outputs/rope_s2_test_n50_r2_ep100.json`。

### 4.7 噪声底与"有效改动"的判定（均在 dev 上）

在 P1 的 E=100 run 上测三类波动：

1. **评估噪声**：同一 ckpt，`--seed` 42 / 43 / 44（同时换起点和 CEM 噪声）。
2. **ckpt 噪声**：同一 run 的 ep90 与 ep100。
3. **训练 seed 噪声**：seed 3072 与 1234。

判定规则：

- 单个环境内，差异必须**大于**（训练 seed 噪声 + 评估噪声），且配对检验 p < 0.05，才算有意义。n=100 时成功率的标准误约 5pp。
- 一个改动要算"有效"，需在 4 个环境中**至少 3 个同向**，其余环境不越过噪声阈值变差。
- 进论文主表的方法补齐 3 个训练 seed（3072 / 1234 / 42）。

## 5. 执行顺序

见改进计划（`s2_冲刺顶会改进计划`）阶段 0：先 P1 → P2/P3 定 E，再补训 C1/S1，然后跑 test 主表 v0。

## 6. 仍待定 / 可选

- **DINO-WM 补充表**（wall/point）：加 `--goal-source random_state`，调用 wrapper 已有的 `sample_random_init_goal_states`。MPC 每轮 5 个动作，上限 20 轮，报告 success@k。只对最终方法和 CLS 跑。
- 权重 EMA（decay 0.999）：如果 ckpt 噪声明显大于评估噪声再加入。
- 评估前重估 BN 统计量：EMA 的替代方案。
- S2 的 SIGReg 把 CLS 和 16 个槽拼在一起算（样本数约为 CLS 的 17 倍）。r2 先照官方 λ=0.09，分组 SIGReg 放到改进计划阶段 2。
- 旧 pickle 与新代码的兼容：后续给 `JEPA` / `ARPredictor` 加新属性时，读取处一律 `getattr(self, name, 旧默认值)`，保证 r2 基线 ckpt 仍能按原行为评估。
