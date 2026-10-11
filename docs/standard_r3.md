# 训练 / 评估标准 r3（2026-09-30 定稿；2026-10-07 更新规划 cost*）

> 取代 [`training_eval_standard_r2.md`](training_eval_standard_r2.md) 及之前所有设定。r2 还没有正式训练过，r3 = r2 + 本文 §2 的修正。  
> r1（2026-09-29 之前）的所有数字只作历史参考，**不与 r3 混表**。  
> 方法相关的 token / pool / pos CLI 仍按 [`S1_S2_eval_protocol.md`](S1_S2_eval_protocol.md) §1.3–1.5；**规划 cost 以本文 §4.2 的 `cost*` 为准**（阶段1在 r3 S2 last 上、dev 贪心扫得到）。

---

## 0. 结论

### 0.1 为什么"改一点结果就疯狂变，四个数据集还各变各的"

按影响从大到小：

1. **point 和 rope 上的模型本身就没学到能用于规划的表征**，所以这两个环境的数字只是随机规划器的波动，改什么都在噪声里乱跳（§1.3）。
   - point：S1 和 S2 的所有 ckpt 都**坍缩**了，编码器对不同帧输出几乎相同的 embedding。原因是 SIGReg 在 17 个 token 展平后的样本上计算，模型可以用"17 个 token 彼此不同、但每个 token 跨帧不变"来满足约束（§2.1）。
   - rope：CLS 也不健康。BatchNorm 训练时只见过 16 个样本，train/eval 两种模式下 embedding 的跨帧标准差相差 10 倍（§2.2）。S2 同样坍缩。
   - wall、granular：CLS 和 S2 都健康，这两个环境的数字可信。
2. **评估样本太少**。rope/granular 的 n=10，Chamfer 的 95% 置信区间宽 0.6–1.6，而方法之间的差距只有 0.05–0.7。wall 上 n=100，S2 100 轮与 150 轮的成功率只差 2 个百分点，但 100 个 episode 里有 26 个成败互换（§1.1）。
3. **r1 训练配方的问题**（r2 文档 §0 已列）：学习率不退火，最后一个 ckpt 停在高学习率上；按 val loss 选出的 best 常落在 ep2/ep4（point S2 150 轮的 best 是 ep2）；梯度累积后没有取平均。
4. **非确定性**（已修）：评估脚本以前没有给 torch 设种子，同一个 ckpt 每次重跑的 CEM 噪声都不同；GPU 算子不确定，同 seed 训练两次结果也不同。

### 0.2 是不是单卡训练的问题

不是卡数的问题，是 r1 **"每步只有 16 个样本"** 这套配方的问题。8 卡机上用的"2 卡 × 16 × 累积 4"，每张卡上 BatchNorm 和 SIGReg 同样只看到 16 个样本，学习率同样不退火，所以问题一样存在。那台机器上训的是 CLS 和 C1，都只有 1 个 token 或不带全局 token，碰不到 S2 的坍缩捷径，因此看起来更稳（这台机器上无法复核那边的结果）。

r3 在单卡上跑**真 batch 128、不累积**：编码器开 gradient checkpointing，峰值显存 6.8 GiB，梯度逐元素不变。这比 8 卡上的 r1 更接近官方配置。

### 0.3 解决方案（按顺序）

| 步 | 做什么 | 状态 |
|---|--------|------|
| 1 | 训练改用 r3：官方超参 + **逐 token SIGReg** + 确定性模式 | 代码已合入，point 上短训练已验证（§2.1） |
| 2 | 每个 ckpt 进规划评估前先跑**健康检查**，不通过的不评估，先查训练 | `scripts/repr_health.py` 已写好，并用已有 ckpt 验证过（§1.3） |
| 3 | 评估用固定协议：split 内全部 episode、固定种子、固定 MPC/CEM 参数 | 脚本默认值已改成协议值（§4） |
| 4 | 先测**噪声底**（训练 seed、评估 seed、ckpt），再判断任何改动 | 待跑，见 §6 的 P1 |
| 5 | 判定规则：配对检验显著 + 超过噪声底 + 4 个环境里至少 3 个同向 | §5 |

---

## 1. 证据：S2 100 轮 vs 150 轮，逐 episode 对比

以下都是 r1 协议（wall MPC 2/10、point 5/10；rope/granular 开环、n=10），同一批 episode。  
分析脚本：`scripts/analysis_r1_noise.py`（读 `outputs/*details*.json` 和 sim 日志里的逐 episode Chamfer）。

### 1.1 wall（n=100，成功率与 Wilson 95% 置信区间）

| ckpt | 成功 | 95% CI |
|------|----:|--------|
| S2 旧 100 轮 best | 79 | 70–86 |
| S2 旧 100 轮 ep100 | 76 | 67–83 |
| **S2 新 150 轮 last** | **78** | 69–85 |
| S2 新 150 轮 best（ep147） | 67 | 57–75 |
| CLS best | 70 | 60–78 |
| CLS ep100 | 65 | 55–74 |

配对 McNemar 检验（"只 A 成功 / 只 B 成功"）：

| 对比 | 只 A | 只 B | p |
|------|----:|----:|---:|
| S2 ep100 vs 150 last | 12 | 14 | 0.85 |
| S2 150 last vs 150 best | 17 | 6 | 0.035 |
| CLS best vs CLS ep100 | 13 | 8 | 0.38 |
| S2 150 last vs CLS best | 18 | 10 | 0.19 |

成功率看起来只差 2 个百分点，但约 **1/4 的 episode 在两个 ckpt 之间成败互换**。在 n=100 下，要差 10 个百分点以上才可能显著。目前所有"S2 赢 CLS"的 wall 结论都**不显著**。

### 1.2 point / rope / granular

| 环境 | 对比 | 结果 |
|------|------|------|
| point | 旧 run 的 ep10–ep60 | 26% → 47% → 35% → 38% → 29% → 33% → 57% → 50%，来回跳；ep50 vs ep59 有 37 个 episode 互换 |
| point | 新 150 轮 ep100 vs last | 36% vs 30% |
| point | CLS best vs CLS ep100 | **63% vs 100%**（ep100 在 7 轮内全部到达，是真实提升） |
| rope | S2 ep100 → 150 last，Chamfer 差 | −0.08，95% CI [−0.79, +0.58] |
| rope | CLS best → CLS ep100 | +0.67，CI [+0.11, +1.33] |
| granular | S2 ep100 → 150 last | +0.05，CI [−0.12, +0.21] |
| granular | CLS best → S2 150 last | +0.23，CI [+0.06, +0.41]（S2 更差，显著） |

rope/granular 只有 10 个 episode，除了个别情况，置信区间都跨过 0，**100 轮与 150 轮之间没有可分辨的差别**。

### 1.3 表征健康检查：谁的数字可信

`scripts/repr_health.py`：在 eval 集的真实轨迹上，取一个终点帧和它之前的 W 帧（wall/point 为 25，rope/granular 为 20），计算"剩余步数"与"当前帧到终点帧的 latent 距离"之间的 Spearman 相关系数 ρ，取 50 条轨迹的平均。终端规划 cost 依赖的正是这个排序；ρ 接近 0 时，规划器无论怎么调参都等于随机选动作。多 token 模型用全部 token 的 ρ。

**判定（按环境）**：跨帧标准差 ≥ 0.05 × max(1, 跨 token 标准差)（防时间坍缩），且 ρ ≥ 环境门槛：

| 环境 | ρ 下限 | 依据 |
|------|-------:|------|
| wall / point_maze | **0.5** | 原标准；健康 run 通常 ≥0.65 |
| granular | **0.45** | 略放宽擦线；规划 Chamfer 已可用 |
| rope | **0.25** | 像素神谕 ρ 仅约 0.27，0.5 不可达；BN 重估后 CLS ≈0.36 视为可用 |

rope **不以 0.5 卡死全流程**。仍要求非坍缩（跨帧 std）；S2 若时间坍缩则仍标 BAD，但已有的规划数字可作 r3 参考基线，并进入后续对比（注明健康状态）。

| 环境 | ckpt | ρ | 跨帧 std | 跨 token std | 健康 | r1 规划结果 |
|------|------|---:|--------:|------------:|:---:|-------------|
| point | CLS best（训练早期） | 0.29 | 0.017 | — | ✗ | 63% |
| point | CLS ep100 | 0.73 | 0.70 | — | ✓ | 100% |
| point | S2 旧 ep3 | 0.27 | 0.0008 | 1.78 | ✗ | 59% |
| point | S2 150 last | 0.33 | 0.0001 | 0.92 | ✗ | 30% |
| wall | CLS ep100 | 0.67 | 0.93 | — | ✓ | 65% |
| wall | S2 150 last | 0.67 | 0.90 | 0.38 | ✓ | 78% |
| rope | CLS best | 0.31 | 0.041 | — | ✗ | Chamfer 1.48 |
| rope | CLS ep100 | 0.30 | 0.006 | — | ✗ | Chamfer 2.15 |
| rope | S2 150 last | 0.08 | 0.0016 | 1.92 | ✗ | Chamfer 2.14 |
| granular | CLS ep100 | 0.66 | 0.59 | — | ✓ | Chamfer 0.64 |
| granular | S2 150 last | 0.65 | 0.62 | 0.64 | ✓ | Chamfer 0.82 |

健康的 ρ 都 ≥ 0.65，不健康的都 ≤ 0.34，0.5 的门槛区分得很干净。另一个独立信号：point 上所有不健康的 ckpt，最终规划 cost 在 100 个 episode 之间的变异系数只有 0.001–0.02，成功和失败的 episode 的 cost 没有差别；健康的 CLS ep100 是 4.3。

**结论**：r1 里 point 和 rope 上的所有 S1/S2 对比、rope 上的 CLS 对比都没有意义。wall 和 granular 的结果可信，但样本量不够下结论。

---

## 2. 根因与修正

### 2.1 S2 坍缩：SIGReg 在展平的 token 上计算

r1/r2 代码：`sigreg_in = rearrange(emb, "b t n d -> t (b n) d")`，即把 17 个 token 和 batch 合在一起做一个分布的高斯检验。

- 如果每个 token 位置输出一个固定向量、跨帧不变，这 17 个向量本身就能凑出一个"还算散开"的分布，SIGReg 惩罚有限；而预测器只需输出常数，`pred_loss` 趋近 0。这是一个很强的局部最优。
- CLS 只有 1 个 token，这条捷径不存在（坍缩会让 SIGReg 爆炸）。
- 实测（训练集帧，batch 128，train 模式）：point S2 的"展平"SIGReg = 81，"逐 token"SIGReg = 128；CLS 约 10。

**修正**：`loss.sigreg.per_token: true`，改为 `"b t n d -> (t n) b d"`：每个 token 位置单独在 batch 维上做 SIGReg，然后对 token 和时间取平均。N=1（CLS）时与官方逐位一致；每个 token 的惩罚尺度与 CLS 相同，所以 λ 沿用官方 0.09。

**验证**：point S2，r2 配方，各训 2 个 epoch（`trainer.max_epochs=2`，其余同 r2），唯一差别是这个开关：

| 变体 | epoch | 跨帧 std | 跨 token std | ρ（全部 token） | val sigreg_loss | 健康 |
|------|-----:|--------:|------------:|---------------:|----------------:|:---:|
| 展平（r2 原样） | 1 | 0.0004 | 1.78 | 0.27 | — | ✗ |
| 展平（r2 原样） | 2 | 0.0004 | 1.67 | 0.21 | 228.5 | ✗ |
| 逐 token（r3） | 1 | 0.77 | 0.077 | 0.25 | — | ✗（还没学到位） |
| 逐 token（r3） | 2 | 0.79 | 0.012 | 0.53 | 8.0 | ✓ |
| 参照：CLS r1 ep10 | 10 | 0.34 | — | 0.72 | — | ✓ |

所以只把 batch 改到 128、λ 改成 0.09（即 r2）**救不了 S2**；逐 token SIGReg 在第 2 个 epoch 就通过了健康检查。

复现：`python train.py --config-name=lewm_point_maze_s2 +recipe=r2 trainer.max_epochs=2 +loss.sigreg.per_token={false|true} subdir=_diag_... wandb.enabled=false save_object_every_n_epochs=1`，再对 `*_epoch_2_object.ckpt` 跑健康检查。

**要持续观察**：逐 token 后，跨 token std 很小（0.012），16 个槽几乎是同一个向量的拷贝。这不影响正确性，但如果完整训练后仍如此，S2 的 16 个槽就没有提供空间信息，m_task cost 会退化成 dense cost。完整训练后要报告跨 token std。这是方法层面的问题，放到改进阶段处理。

### 2.2 rope CLS：BatchNorm 只见过 16 个样本

projector 和 pred_proj 里有 `BatchNorm1d`。r1 每步 16 个样本，模型学到依赖小 batch 统计量的特征。实测 rope CLS ep100：

| 模式 | 跨帧 std | SIGReg |
|------|--------:|------:|
| eval（running stats，规划时用） | 0.019 | 405 |
| train（batch stats） | 0.175 | 46 |

wall/point/granular 的 CLS 在两种模式下基本一致（SIGReg 4–11）。r3 用真 batch 128（与官方相同），应能消除这个问题；rope CLS 在 r3 下的健康检查是 P1 的第一道关卡（§6）。

### 2.3 已修的评估端问题（2026-09-29 至 30）

| 问题 | 影响范围 | 修法 |
|------|---------|------|
| CEM 没有 torch 种子 | 09-29 之前所有评估 | 每个 episode 开始时 `torch.manual_seed(seed + i)` |
| `eval_wall.py` 默认 `--plan-gpu 2`，这台机器只有 0/1 两张卡 → 静默退回 CPU | 没显式传 `--plan-gpu` 的 wall 评估（正式脚本都传了 0/1，不受影响） | 默认 0；CUDA 不可用时直接报错，不再退回 CPU |
| `eval_point_maze.py` 默认 `--plan-gpu 7` | 同上（直接报错） | 默认 0 |
| CLI 无条件覆盖 ckpt 里的 `inject_every_layer` / `n_lo_mode` | 评估 S2 时漏传 `--inject-every-layer` 就会悄悄改变结构（正式脚本都传了，不受影响） | 只在显式传参时覆盖，否则用 ckpt 里的值 |
| rope/granular 的 plan 文件固定路径 | 并行评估会互相覆盖 | 默认跟随 `--output` |
| `--split dev` 只有 96 个 wall episode，`--num-eval 100` 会有放回抽样 | 重复 episode | `--num-eval 0`（新默认）= split 内每个 episode 恰好一次 |
| `eval_granular.py` 默认 `planner=mpc`（3/8） | 漏传就跑成 MPC | 默认改为 `openloop` |
| rope/granular 结果只有汇总值，逐 episode Chamfer 只在日志里 | 做不了配对检验 | 仿真 worker 输出 `details`；另写 `<output>_metrics.json` |
| wall/point 脚本默认 MPC 3/8、5/5 | 漏传就是非协议设定 | 默认 5/10 |

---

## 3. 训练标准 r3

配方：`config/train/recipe/r3.yaml`（继承 `r2.yaml`，另加 `trainer.deterministic` 和 `loss.sigreg.per_token`）。四个环境、所有方法（CLS / C1 / S1 / S2）统一。

| 项 | r3 | 官方 LeWM（commit 83f97d7） | 说明 |
|----|----|----------------------------|------|
| 全局 batch | **128，不累积** | 128，不累积 | |
| 设备 | 单卡 `[0]` | 单卡 | |
| 精度 | bf16-mixed（SIGReg 也在 bf16 下） | bf16 | |
| 优化器 | AdamW，lr **5e-5**，wd **1e-3** | 同 | 取代 r1 各环境改过的 lr 3e-5 / wd 2e-3 |
| LR 调度 | 1% 线性 warmup + 按优化器步余弦到 0 | 同 | r1 实际不退火 |
| grad clip | 1.0 | 1.0 | |
| SIGReg | λ **0.09**，knots 17，num_proj 1024 | 同 | 取代 r1 的 λ 0.05 |
| SIGReg 多 token | **逐 token**（`per_token: true`） | 官方只有 CLS；N=1 时逐位相同 | §2.1 |
| 编码器 | ViT-tiny，patch 14，224×224 | 同 | |
| predictor | depth 6，heads 16，mlp 2048，dim_head 64，dropout 0.1，emb_dropout 0 | 同 | |
| history / num_preds / embed_dim | 3 / 1 / 192 | 同 | |
| frameskip | wall/point 5，rope/granular 1 | 由数据决定 | DINO-WM 数据集设定 |
| train / val 划分 | 0.9 / 0.1，`random_split` 用 seed 生成器 | 同 | |
| max_epochs | **100** | 代码默认 100；DINO-WM 在这四个数据集上也训 100 | |
| seed | **3072**（噪声底另加 1234，最终表另加 42） | 3072 | |
| 确定性 | `trainer.deterministic: true` | 未开 | 同 seed 逐位可复现，约慢 10% |
| 编码器 grad ckpt | 开 | 关 | 数学等价，只省显存 |
| 正式权重 | **`*_last_object.ckpt`**（= epoch_100） | — | 不用按 val loss 选的 best |
| 存档 | 每 10 epoch 一个 object + last + `*_weights.ckpt`（续训用） | — | |

每 epoch 步数（batch 128）：wall ≈ 376，point ≈ 1025，rope/granular ≈ 113。

```bash
export STABLEWM_HOME=/data/czx/le-wm/first/data
export CUDA_VISIBLE_DEVICES=0
# 正式入口已烘焙 r3，无需 +recipe=r3 / 无需手改开关：
scripts/train_r3.sh rope cls                 # CLS
scripts/train_r3.sh rope s2                  # S2
scripts/train_r3.sh rope s2 seed=1234        # 噪声底
# 等价: .venv/bin/python train.py --config-name=r3/rope_s2
# C1 / S1：scripts/train_r3.sh <env> c1|s1
```

- 配置目录：`config/train/r3/<env>_<method>.yaml`（见该目录 README）。
- run 目录：`data/runs/r3/<env>_<method>_s<seed>_<时间戳>/`，里面有完整的 `config.yaml`。
- 中断续训：用同一个 `subdir=runs/r3/<原目录名>` 重跑，自动读 `*_weights.ckpt`（优化器、调度器、RNG 一起恢复）。
- 不要用一个长 run 的中间 ckpt 代替短 run（余弦调度由总步数决定），见 r2 文档 §3.2。
- 评估可直接：`scripts/eval_r3.sh <env> <method> <ckpt>`（方法 CLI 已内置）。

### 3.1 训练过程中要看的

- `fit/sigreg_loss`：CLS 和 S2 应在同一量级（约 5–15）。S2 如果比 CLS 高一个数量级（r1 point S2 为 80–230），说明在坍缩。
- 每 10 个 epoch 的 object ckpt 可以随时跑健康检查（几十秒），不用等训练结束。

---

## 4. 评估标准 r3

### 4.1 进评估前：健康检查（必做）

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/repr_health.py --env <wall|point_maze|rope|granular> \
    --ckpt data/<run>/<name>_last_object.ckpt --json outputs/r3/<env>_<method>_s<seed>_health.json
```

输出 `[OK ]` 才进入规划评估。`[BAD]` 的 ckpt 不评估，先查训练；它的规划数字没有意义。

### 4.2 协议（一套规则，四个数据集共用）

四个数据集都来自 DINO-WM，规划设定按 DINO-WM 的 plan 配置；导航任务的执行预算按 LeWM 论文（预算 = 2 × 目标距离）。

| 项 | wall | point_maze | rope | granular | 依据 |
|----|------|-----------|------|----------|------|
| 目标 | 数据集轨迹 +25 环境步 | +25 | +5 | +5 | DINO-WM `goal_H`=5 × frameskip |
| 规划器 | MPC | MPC | 开环 CEM | 开环 CEM | DINO-WM：`plan_wall`/`plan_point_maze` 为 MPC，`plan.yaml`+`planner/cem` 为开环 |
| horizon | 5 | 5 | 5 | 5 | DINO-WM / LeWM |
| CEM 样本 / top-k / var | 300 / 30 / 1.0 | 同 | 同 | 同 | DINO-WM `cem.yaml`，LeWM `solver/cem.yaml` |
| CEM 迭代 | **10** | **10** | **30** | **30** | DINO-WM：MPC 子规划 `opt_steps` 10，开环 30 |
| 每轮执行 | 5 个 macro（整段 horizon） | 5 | — | — | DINO-WM `n_taken_actions` 5 = LeWM `receding_horizon` 5 |
| MPC 最多轮数 | 10 | 10 | — | — | |
| 规划上下文 | 只用当前帧（`--plan-protocol lewm`） | 同 | 同 | 同 | LeWM `history_len`=1，DINO-WM `obs_0` |
| cost | CLS `dense_terminal`；C1/S1/S2 用 **`cost*`**（§4.2.1） | 同 | 同 | 同 | 阶段1 dev 扫；旧 `m_task_terminal` 仅作消融对照 |
| 成功判定 | ‖Δxy‖ < 4.5 | ‖Δxy‖ < 0.5 | Chamfer < 1.0 | Chamfer < 1.0 | DINO-WM |
| **主指标** | **success@2**（50 环境步） | **success@2** | **Chamfer 均值**（另报中位数） | **Chamfer 均值** | LeWM 预算 50；DINO-WM 可变形体用 Chamfer |
| 副指标 | success@10 | success@10 | 成功率（附录） | 成功率（附录） | |
| episode | `--split dev` 用于开发；`--split test` 只用于最终表 | 同 | 同 | 同 | 奇数号 / 偶数号 episode，互不重叠 |
| n | `--num-eval 0`（全 split）= **96** | **100** | **50** | **50** | 每个 episode 恰好一次 |
| seed | 42；每个 episode `torch.manual_seed(42 + i)` | 同 | 同 | 同 | |
| GPU | 规划 GPU0 | 规划 GPU0，渲染要先 `source setup_env.sh` | 规划 GPU0，EGL 渲染 GPU1 | 同 rope | |

说明：

- wall/point 只跑一次 `--mpc-max-iters 10`，用 `scripts/mpc_budget_curve.py` 读 success@k。每个 episode 单独设种子，所以前 k 轮与单独跑 `--mpc-max-iters k` 逐 episode 相同。
- wall/point 同时报告起点已在成功球内（`mpc_iters=0`）的 episode 数，各方法相同。
- 开环就是 MPC 的第 1 轮：rope/granular 的 g=5、fs=1 时，一次 5 步规划正好覆盖目标距离。
- r1 的 wall 2/10、point 5/10 只作附录里的规划器敏感性对照，不进主表。
- rope 只用 `eval_rope_1.py`；旧的 `eval_rope.py` 默认 g15、CEM 128/5/16，不要用。

#### 4.2.1 规划 cost*（2026-10-07，阶段1）

在四环境 r3 S2 `*_last_object.ckpt` 上、**只看 `--split dev`** 贪心扫得到；门控：≥3 个环境相对上一步变好才采纳。最终：

| 项 | 值 | 说明 |
|----|----|------|
| `--cost-mode` | **`decoupled`** | \(C=\hat c_{\mathrm{CLS}}+\lambda\cdot\hat c_{\mathrm{slot}}\) |
| `--cost-norm` | **`calib`** | 组内尺度用候选中位数标定 |
| `--cost-lambda` | **`2`** | 槽相对 CLS |
| `--mask-weight` | **`hard`** | soft / T 扫描未过门控 |
| `--tau-mode` | **`fixed`** | quantile / otsu 未过门控 |
| `--m-task-threshold` | **`0.08`** | 与旧 eps 阈值相同 |
| `--relevance-source` | **`pixel`** | attn_proj / latent 未过门控 |
| `--cvar-q` / `--cvar-beta` | **`0.1` / `0.5`** | token-CVaR |
| `--lambda-anneal` | **`off`** | 线性退火未过门控 |

CLS 仍用 `dense_terminal`（单 token，无槽）。C1 / S1 / S2 正式评估一律用上面的 `cost*`；旧 `m_task_terminal` 只作消融。  
dev 上 S2：wall success@2 38.54%→41.67%（与 CLS 持平），rope Chamfer 1.973→1.755（优于 CLS 1.822），point 持平 99%，granular 略差（0.704→0.715，仍落后 CLS 0.670）。结果目录：`outputs/r3/p1_cost/`。

### 4.3 命令

```bash
cd /data/czx/le-wm/first
source setup_env.sh                      # point / rope / granular 需要（MuJoCo、PyFleX、EGL）
export STABLEWM_HOME=$PWD/data
CK=data/<run>/<name>_last_object.ckpt
TAG=<env>_<method>_s<train_seed>_dev_e42   # 最终表把 dev 换成 test
# 规划 cost*（C1/S1/S2 共用；见 §4.2.1）
COST_STAR="--cost-mode decoupled --cost-norm calib --cost-lambda 2 --mask-weight hard \
  --tau-mode fixed --m-task-threshold 0.08 --relevance-source pixel \
  --cvar-q 0.1 --cvar-beta 0.5 --lambda-anneal off"
# 方法参数（表示与训练一致；cost 用 cost*）：
CLS="--token-mode cls --cost-mode dense_terminal"
S2="--token-mode spatial16 --pool-mode query --include-global-token --pos-mode nibble --inject-every-layer --n-lo-mode attn $COST_STAR"
S1="--token-mode spatial16 --pool-mode query --include-global-token --pos-mode learned $COST_STAR"
C1="--token-mode spatial16 --pool-mode avg $COST_STAR"
M=$S2   # 按方法选择

# wall / point
.venv/bin/python eval_wall.py --ckpt $CK --dataset data/wall_eval.h5 --split dev --num-eval 0 --seed 42 \
  --goal-offset 25 --horizon 5 --planner mpc --n-taken-actions 5 --mpc-max-iters 10 \
  --cem-num-samples 300 --cem-topk 30 --cem-n-steps 10 --plan-protocol lewm --plan-gpu 0 $M \
  --output outputs/r3/$TAG.txt --details-json outputs/r3/${TAG}_details.json
# point：eval_point_maze.py，--dataset data/point_maze_eval.h5，其余参数相同

# rope / granular
.venv/bin/python eval_rope_1.py --ckpt $CK --dataset data/rope_eval.h5 --split dev --num-eval 0 --seed 42 \
  --goal-offset 5 --horizon 5 --history-size 3 --action-dim 4 \
  --cem-num-samples 300 --cem-topk 30 --cem-n-steps 30 --plan-protocol lewm \
  --plan-gpu 0 --egl-gpu 1 $M --output outputs/r3/$TAG.txt 2>&1 | tee outputs/r3/$TAG.log
# granular：eval_granular.py，--dataset data/granular_eval.h5 --planner openloop，其余相同
# 输出：$TAG.txt（汇总）+ ${TAG}_metrics.json（汇总 + 逐 episode 的 ep_id / start_step / chamfer / success）

# 读 wall/point 预算曲线（第一个文件为基线，其余逐个与它做配对 McNemar）
.venv/bin/python scripts/mpc_budget_curve.py outputs/r3/wall_cls_..._details.json outputs/r3/wall_s2_..._details.json \
  --frameskip 5 --n-taken 5 --ks 1 2 4 10
```

结果统一放在 `outputs/r3/`，文件名带环境、方法、训练 seed、split、评估 seed。

耗时取决于成功得多早（成功即停）。r1 的实测上限：wall S2 n=100 约 11 分钟；point S2（几乎全部跑满 10 轮）n=100 约 28 分钟，point CLS ep100 只要 70 秒。rope 1 个 episode 的冒烟测试约 1 分钟，其中大部分是 PyFleX 初始化。

---

## 5. 噪声底与"有效改动"的判定

### 5.1 三类噪声（P1 时测）

| 噪声 | 怎么测 |
|------|-------|
| 评估噪声 | 同一 ckpt，`--seed 42 / 43 / 44`（同时换起点和 CEM 噪声） |
| ckpt 噪声 | 同一 run 的 `epoch_90` vs `last` |
| 训练 seed 噪声 | 同一配置 `seed=3072` vs `seed=1234`（同 seed 已逐位可复现，不需要重复跑） |

### 5.2 规则

1. **两个 ckpt 原则上都要通过健康检查**再比；**rope** 用环境专用 ρ 门槛（§1.3），不以 0.5 卡死。时间坍缩（跨帧 std 过低）的 ckpt 规划数字只作参考，方法对比时在表里标注。
2. 同一环境内：配对检验显著（成功率用 McNemar p < 0.05；Chamfer 用配对 bootstrap 95% CI 不含 0），**并且**差值大于同一环境实测的训练 seed 噪声。参考量级：n≈100 时，成功率要差约 10 个百分点才可能显著。
3. 一个改动要算"有效"：4 个环境里**至少 3 个同向**，其余环境没有显著变差。
4. 所有开发和调参都在 `dev` split 上做；`test` split 只在最终定稿时跑一次。
5. 进论文主表的方法补齐 3 个训练 seed（3072 / 1234 / 42），报告跨 seed 的均值 ± 标准差（LeWM 论文同样是 3 seed）。
6. 一律用 `last`，不用按 val loss 选的 best。

---

## 6. 执行顺序

GPU0 一次只放一个训练（GPU1 经常被他人占满，只给 rope/granular 当 EGL 渲染卡）。以下耗时是 GPU0 独占、确定性模式下的估计。

| 阶段 | 内容 | 训练耗时 | 通过条件 |
|------|------|---------|---------|
| **P1** | rope、granular × {CLS, S2}（seed 3072 先出基线；噪声底另加 1234） | 每环境约 2×1 小时 | 按 §1.3 环境门槛；**rope 不以 0.5 卡死**；已有 dev 规划数字可作 r3 参考基线 |
| P2 | wall × {CLS, S2}，seed 3072 | 2 × 约 3 小时 | 健康检查通过 |
| P3 | point × {CLS, S2}，seed 3072 | 2 × 约 8 小时 | point S2 健康检查通过（r1 下坍缩的环境） |
| P4 | 四环境补 C1、S1 | — | 得到 r3 版 CLS / C1 / S1 / S2 完整基线 |
| P5 | S2 改进：阶段1 cost* **已定**（§4.2.1） | — | |
| P5b | rope 上重训 C2（平衡 loss）/ C3（locality·continuous·pos_prop） | 各约 6 变体 | **全部未过门控**（相对 cost* 基线变差或 n.s.）；表征健康检查亦多为 BAD |

P1–P3 现状：四环境 CLS/S2（seed 3072）已训完；阶段1 零重训 cost 扫已完成。

**当前最优方案（2026-10-09 冻结）**：r3 S2 `*_last_object.ckpt` + §4.2.1 `cost*`（不叠 C2/C3）。正式主表与后续多 seed / test 均以此为准；C2/C3 仅作负结果附录。结果目录：`outputs/r3/p1_cost/`（cost*）、`outputs/r3/p2_loss/`、`outputs/r3/p3_repr/`。

---

## 7. 复现清单

每个正式结果都要能回溯到：

- run 目录下的 `config.yaml`（完整 Hydra 配置，含 seed、recipe、`per_token`、`deterministic`）；
- 用的 ckpt 文件名（`*_last_object.ckpt`）；
- 健康检查 JSON；
- 评估命令行（结果 txt/json 里的 metrics 段记录了 horizon、MPC、goal_offset 等）；
- 逐 episode 明细（wall/point 的 `--details-json`，rope/granular 的 `*_metrics.json`）。

rope/granular 的方法间配对比较：`scripts/paired_chamfer.py base_metrics.json other_metrics.json ...`（配对 bootstrap，10k 次）。wall/point 用 `scripts/mpc_budget_curve.py`（配对 McNemar）。

已知的剩余非确定性：PyFleX 仿真的 Chamfer 在重跑时有约 1e-4 的差异，远小于方法间差距，可以忽略。

---

## 8. 与 r2 的差异

| 项 | r2 | r3 |
|----|----|----|
| SIGReg（多 token） | 展平 `t (b n) d` | 逐 token `(t n) b d` |
| `trainer.deterministic` | 开（recipe） | 开 |
| 正式入口 | `lewm_*` + `+recipe=r2` | **`config/train/r3/<env>_<method>`**（已烘焙，无需 +recipe） |
| run 目录 | `runs/r2/...` | `runs/r3/<env>_<method>_s<seed>_<时间>` |
| 评估前健康检查 | 无 | 必做 |
| 评估 n | wall/point 100，rope/granular 50 | 全 split：96 / 100 / 50 / 50，每个 episode 一次 |
| granular 脚本默认 planner | mpc | openloop |

CLS 模型（单 token）在 r2 和 r3 下的训练目标完全相同，区别只在确定性模式与入口路径。
