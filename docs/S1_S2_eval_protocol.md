# S1 / S2 四数据集评估协议与结果汇总

> **2026-09-30：时程、MPC、n、split、种子以 [`standard_r3.md`](standard_r3.md) 为准。** 本文 §2–§3 均为 r1 结果；其中 point 上的 S1/S2 和 rope 上的所有方法都没通过表征健康检查（r3 文档 §1.3），这些数字不能用来比较方法。本文 §1.3–1.5 的方法 CLI（token / pool / pos / cost）仍然有效。

> 更新时间：2026-09-29  
> 范围：wall / point_maze / rope / granular 上 **CLS / S1 / S2（norel）** 的评估协议与结果。  
> 记号：MPC `a/b` = `--n-taken-actions a --mpc-max-iters b`。  
> 标准训练 run / ckpt：[`standard_training_checkpoints.md`](standard_training_checkpoints.md)。  
> 文档结构：**§1 评估标准（详细）→ §2 正式主表结果 → §3 消融/对照（历史全保留，含末 epoch）→ §4 配置背景 → §5 文件索引**。  
> **未删任何已有数字**：短程、旧 MPC、epoch 扫档、cost 消融、早期非 norel S2、旧 100ep 末 epoch、**CLS 末 epoch** 等均在 §3。  
> **S2 现行正式权重**（2026-09-28）：四数据集 **150ep `*_last_object.ckpt`**；§2 主表 S2 列已换成该次正式协议结果。

---

## 1. 评估标准（CLS / S1 / S2 各用什么）

### 1.1 一句话对照

| 方法 | 表示（重建权重时必须对上） | 规划 cost | wall/point 时程与 MPC | rope/granular |
|------|---------------------------|-----------|----------------------|---------------|
| **CLS**（原版 LeWM） | `--token-mode cls`（单 CLS；不要传 query / global / nibble / inject） | **`dense_terminal`** | 与下表同：g25 + wall **2/10** / point **5/10** | 开环 CEM，goal=5 |
| **S1** | `spatial16` + `--pool-mode query` + `--include-global-token`；`--pos-mode learned`；**无** nibble / inject | **`m_task_terminal`** | 同上 | 同上 |
| **S2 norel** | 同 S1 读出，另加 `--pos-mode nibble --inject-every-layer --n-lo-mode attn` | **`m_task_terminal`** | 同上 | 同上 |

要点：

- **时程 / planner / 成功球** 三方法共用正式约定；**表示与 cost 必须跟训练一致**。  
- CLS **不能**用 `m_task`（那就不是原版）；S1/S2 主表 **不能**漏传而落到脚本默认的 `dense`（那就变成消融）。  
- C1（avg pool + `m_task`）协议与 S1/S2 主表的 cost/时程相同，但表示不同；本文件主表聚焦 CLS/S1/S2。

### 1.2 按数据集：正式时程 / planner / 指标

> **r2 起（2026-09-29）**：wall/point 改为跑 `5/10`，主指标 success@2（50 环境步，LeWM 预算），副栏 success@10；评估脚本已改为每个 episode 单独设 torch 种子，并新增 `--split dev|test`（调参只用 dev）。n、划分、rope 用哪个脚本等，均以 [`training_eval_standard_r2.md`](training_eval_standard_r2.md) §4 为准。下表 MPC 2/10、5/10 仅对应 r1 历史结果。

| 数据集 | frameskip | goal_offset | planner | MPC | 主指标 | 成功阈值 | num-eval（正式习惯） |
|--------|-----------|-------------|---------|-----|--------|----------|----------------------|
| **wall** | 5 | **25**（= DINO `goal_H`×fs） | **MPC** | **2/10** | 成功率 + mean/median `state_dist` | `‖Δxy‖ < 4.5` | **100** |
| **point_maze** | 5 | **25** | **MPC** | **5/10** | 同上 | `‖Δxy‖ < 0.5` | **100** |
| **rope** | 1 | **5** | **开环 CEM**（无 MPC） | — | **Chamfer** mean/median（成功率仅附录） | Chamfer `< 1.0` | ≥50 更好；现有常 **10** |
| **granular** | 1 | **5** | **开环 CEM** | — | **Chamfer** mean/median | Chamfer `< 1.0` | ≥50 更好；现有常 **10** |

公共：

- `horizon=5`，`history_size=3`。  
- CEM 内层：wall/point `cem-n-steps=10, samples=300, topk=30`；rope/granular `cem-n-steps=30`（同默认）。  
- **禁止** wall/point 用 `goal_offset=5` 做正式对比（fs=5 时仅 1 个 macro，大量 `mpc_iters=0` 虚高）。详见 §1.6。

### 1.3 CLS 评估标准（详细）

训练：`config/train/lewm_{wall,point_maze,rope,granular}.yaml`（无 `_c1/_s1/_s2`），缺省 `token_mode=cls`，规划 `dense_terminal`。

评估 CLI 要点：

```bash
--token-mode cls \
--cost-mode dense_terminal
# 不要传: --pool-mode query / --include-global-token / --pos-mode nibble / --inject-every-layer
# 不要用 m_task_terminal
```

| 数据集 | 完整正式设定 |
|--------|--------------|
| wall | g25，MPC **2/10**，`dense_terminal`，`cls`，n=100，fs=5 |
| point | g25，MPC **5/10**，`dense_terminal`，`cls`，n=100，fs=5 |
| rope | g5，**开环 CEM**，`dense_terminal`，`cls`，n≥10（建议≥50） |
| granular | g5，**开环**（`--planner openloop`），`dense_terminal`，`cls`，n≥10 |

标准 ckpt 见 [`standard_training_checkpoints.md`](standard_training_checkpoints.md)：  
wall `lewm_wall_v2_20260922_135024`；point `lewm_point_maze_v2_20260922_193111`；  
rope `lewm_rope_v3_20260922_154251`；granular `lewm_granular_v1_20260922_193208`。

### 1.4 S1 评估标准（详细）

训练：`lewm_*_s1.yaml`。读出 = query + global CLS；PE = learned；无 NibblePE / 无层注入。

```bash
--token-mode spatial16 --pool-mode query --include-global-token \
--pos-mode learned \
--cost-mode m_task_terminal
# 不要传: --pos-mode nibble / --inject-every-layer
```

| 数据集 | 完整正式设定 |
|--------|--------------|
| wall | g25，MPC **2/10**，`m_task_terminal`，n=100 |
| point | g25，MPC **5/10**，`m_task_terminal`，n=100；ckpt 用正式档 **`best_ep8`** |
| rope / granular | g5，开环，`m_task_terminal`，n≥10 |

`m_task` 默认：`--m-task-mode eps --m-task-threshold 0.08`（与 C1 yaml 一致）。

### 1.5 S2（norel）评估标准（详细）

训练：`lewm_*_s2.yaml`（norel：`pos_mode=nibble`，`inject_every_layer`，`use_relative_bias=false`）。

```bash
--token-mode spatial16 --pool-mode query --include-global-token \
--pos-mode nibble --inject-every-layer --n-lo-mode attn \
--cost-mode m_task_terminal
```

| 数据集 | 完整正式设定 |
|--------|--------------|
| wall | g25，MPC **2/10**，`m_task`，n=100；ckpt 见 [`standard_training_checkpoints.md`](standard_training_checkpoints.md)（现行 **150ep last**） |
| point | g25，MPC **5/10**，`m_task`，n=100；ckpt 同上（现行 **150ep last**） |
| rope | g5，开环，`m_task`，n≥10；ckpt 同上（现行 **150ep last**） |
| granular | g5，开环，`m_task`，n≥10；ckpt 同上（现行 **150ep last**） |

主表 S2 **锁 16 槽 + NibblePE**；把 cost 改成 `dense_terminal` 属于 **消融**（§3），不是正式主表。

### 1.6 `goal_offset` 与 frameskip

```
raw env steps (goal_offset) ÷ frameskip = macro 步数（应对齐 DINO goal_H=5）
```

| 数据集 | fs | 正确 goal_offset | 错误短程 |
|--------|----|------------------|----------|
| wall / point | 5 | **25** | 5（=1 macro） |
| rope / granular | 1 | **5** | — |

为何 wall/point 不能用 goal=5：开局常已在成功球内 → `mpc_iters=0` 虚高；调大 `mpc_max_iters` 几乎不动 headline。正式对比只看 g25。

### 1.7 脚本默认值陷阱

| 脚本 | CLI 默认（易踩坑） | 正式应显式传 |
|------|-------------------|--------------|
| `eval_wall.py` | MPC 常为 3/8；cost 默认 dense | `--n-taken-actions 2 --mpc-max-iters 10`；S1/S2 再加 `m_task` |
| `eval_point_maze.py` | MPC 常为 5/5；cost 默认 dense | `--n-taken-actions 5 --mpc-max-iters 10`；S1/S2 加 `m_task` |
| `eval_granular.py` | 默认 `planner=mpc` | 正式一律 `--planner openloop` |
| `eval_rope_1.py` | 仅开环（无 mpc 开关） | 传齐 token/cost；n 建议加大 |

---

## 2. 正式主表结果（CLS / S1 / S2）

下列为**各方法正式协议**下的 headline。rope/granular 主文看 **Chamfer**；成功率因 n=10 很噪，括号附上。

### 2.1 总览

| 数据集 | 协议摘要 | CLS | S1 | S2 norel（主表，**150ep last**） |
|--------|----------|-----|----|----------------------------------|
| **wall** | g25，MPC 2/10；CLS=`dense`；S1/S2=`m_task`；n=100 | **70%**（7.16 / 3.31） | **71%**（6.66 / 3.59） | **78%**（5.56 / 3.35） |
| **point** | g25，MPC 5/10；cost 同上；n=100 | **63%**（3.05 / 2.85） | **25%**（3.50 / 3.35） | **30%**（3.36 / 3.39） |
| **rope** | 开环 g5；n=10；主指标 Chamfer | **20%**；**1.48 / 1.26** | **10%**；**1.93 / 1.91** | **30%**；**2.14 / 1.98** |
| **granular** | 开环 g5；n=10；主指标 Chamfer | **100%**；**0.59 / 0.68** | **80%**；**0.84 / 0.69** | **80%**；**0.82 / 0.75** |

括号内 wall/point 为 mean / median `state_dist`；rope/granular 为 mean / median Chamfer。  
旧 S2 主表（100ep best / point planbest=ep3）数字保留在 §3.7，勿与上表混报。

### 2.2 Wall（正式）

| 方法 | 结果文件 | cost | success | mean / median state_dist |
|------|----------|------|---------|--------------------------|
| CLS | `outputs/wall_cls_results_n100_g25_t2i10.txt` | dense | **70%** | 7.16 / 3.31 |
| S1 | `outputs/wall_s1_results_n100_g25_t2i10.txt` | m_task | **71%** | 6.66 / 3.59 |
| **S2 norel（150ep last）** | `outputs/wall_s2_results_n100_g25_t2i10_ep150.txt` | m_task | **78%** | 5.56 / 3.35 |

权重：CLS `lewm_wall_v2_20260922_135024`；S1 `lewm_wall_s1_20260917_104917`；  
S2 `lewm_wall_s2_20260927_173022/..._last_object.ckpt`。

### 2.3 Point Maze（正式）

| 方法 | 结果文件 | cost | success | mean / median state_dist |
|------|----------|------|---------|--------------------------|
| CLS | `outputs/point_maze_cls_results_n100_g25_t5i10.txt` | dense | **63%** | 3.05 / 2.85 |
| S1（ep8） | `outputs/point_maze_s1_results_n100_g25_t5i10.txt` | m_task | **25%** | 3.50 / 3.35 |
| **S2 norel（150ep last）** | `outputs/point_maze_s2_results_n100_g25_t5i10_ep150.txt` | m_task | **30%** | 3.36 / 3.39 |

权重：CLS `lewm_point_maze_v2_20260922_193111`；S1 `..._s1_20260920_193429/..._best_ep8_object.ckpt`；  
S2 `..._s2_20260927_173356/..._last_object.ckpt`。  
（point `state_dist` 为全状态范数含速度，可与 XY 成功脱节。）

### 2.4 Rope（正式；n=10，Chamfer 为主）

| 方法 | 结果文件 | cost | success | mean / median Chamfer |
|------|----------|------|---------|------------------------|
| CLS | `outputs/rope_cls_metrics_n10.json` | dense | **20%** | **1.48 / 1.26** |
| S1 | `outputs/rope_s1_best_metrics_n10.json` | m_task | **10%** | **1.93 / 1.91** |
| **S2 norel（150ep last）** | `outputs/rope_s2_norel_metrics_n10_ep150.json` | m_task | **30%** | **2.14 / 1.98** |

权重：CLS `lewm_rope_v3_20260922_154251`；S1 `lewm_rope_s1_20260916_201520`；  
S2 `lewm_rope_s2_20260926_142449/..._last_object.ckpt`。

### 2.5 Granular（正式；n=10，Chamfer 为主）

| 方法 | 结果文件 | cost | success | mean / median Chamfer |
|------|----------|------|---------|------------------------|
| CLS | `outputs/granular_cls_metrics_n10.json` | dense | **100%** | **0.59 / 0.68** |
| S1 | `outputs/granular_s1_best_metrics_n10.json` | m_task | **80%** | **0.84 / 0.69** |
| **S2 norel（150ep last）** | `outputs/granular_s2_norel_metrics_n10_ep150.json` | m_task | **80%** | **0.82 / 0.75** |

权重：CLS `lewm_granular_v1_20260922_193208`；S1 `lewm_granular_s1_20260917_083958`；  
S2 `lewm_granular_s2_20260926_161822/..._last_object.ckpt`。

---

## 3. 消融与对照结果（历史全保留，勿与主表混列）

> 本节收录：短程 goal=5、非正式 MPC、开环对照、S2 cost=`dense` 消融、point 多 epoch 扫档、S2 末 epoch、早期非 norel S2、C1 零散结果等。

### 3.1 Wall — 短程 / 旧 MPC / cost 消融 / early 拆解

| 标签 | 结果文件 | goal | MPC | cost | n | success | mean / median dist |
|------|----------|------|-----|------|---|---------|---------------------|
| S1 短程 3/8 | `wall_s1_best_results_n100.txt` | 5 | 3/8 | m_task | 100 | 92% | 3.37 / 3.30 |
| S1 短程 2/10 | `wall_s1_results_n100_t2i10.txt` | 5 | 2/10 | m_task | 100 | 95% | 3.29 / 3.37 |
| S2 旧（含 rel 等）短程 3/8 | `wall_s2_best_results_n100.txt` | 5 | 3/8 | m_task | 100 | 84% | 5.83 / 3.64 |
| S2 norel 短程 3/8 | `wall_s2_norel_results_n100.txt` | 5 | 3/8 | m_task | 100 | 90% | 3.73 / 3.26 |
| S2 norel 短程 2/10 | `wall_s2_norel_results_n100_t2i10.txt` | 5 | 2/10 | m_task | 100 | 95% | 3.00 / 3.30 |
| **S2 norel cost 消融** | `wall_s2_results_n100_g25_t2i10_dense.txt` | **25** | **2/10** | **dense** | 100 | **73%** | 6.90 / 3.39 |

**early-success**（seed=42；`mpc_iters=0` = 未进 CEM）：

| 设定 | early/100 | early 且成功 | 需规划子集成功 |
|------|-----------|--------------|----------------|
| S1/S2 goal=5 任意 MPC | 74 | 74 | S1 3/8: 18/26；S1 2/10: 21/26；S2 3/8: 16/26；S2 2/10: 21/26 |
| S1 g25 2/10 m_task | 26 | 26 | **45/74** |
| S2 norel g25 2/10 m_task（主表） | 26 | 26 | **53/74** |
| S2 norel g25 2/10 dense（消融） | 26 | 26 | **47/74** |

解读：短程 headline 92–95% 中约 3/4 是开局已到；正式看 g25。S2 换 dense 从 79%→73%，仍略高于 CLS 70%，但不如 m_task。

### 3.2 Point — 短程 / 非正式 MPC / epoch 扫档 / cost 消融

| 标签 | 结果文件 | goal | MPC | cost | success | mean / median dist |
|------|----------|------|-----|------|---------|---------------------|
| S2 误用 ep45 短程 5/5 | `point_maze_s2_results_n100.txt` | 5 | 5/5 | m_task | 61% | 3.12 / 2.85 |
| S2 同上短程 5/10 | `point_maze_s2_results_n100_t5i10.txt` | 5 | 5/10 | m_task | 62% | 3.11 / 2.83 |
| S2 ep3 历史 5/5 | `point_maze_s2_results_n100_c1proto.txt` | 25 | **5/5** | m_task | 50% | 3.40 / 3.33 |
| S2 ep3 开环对照 | `point_maze_s2_results_n100_g25_openloop_ep3.txt` | 25 | openloop | m_task | （见该文件） | — |
| S1 开环对照 | `point_maze_s1_results_n100_g25_openloop.txt` | 25 | openloop | m_task | （见该文件） | — |
| **S2 ep3 cost 消融** | `..._g25_t5i10_ep3_dense.txt` | **25** | **5/10** | **dense** | **61%** | 3.46 / 3.42 |

**early（goal=5）**：约 58/100 开局已成功 → 短程 ~61% 几乎不测规划。  
**ep3 g25 5/5**：early 15；规划子集 35/85 → 50%。  
**ep3 g25 5/10 m_task（主表）**：early 15；规划 **44/85** → 59%。  
**ep3 g25 5/10 dense**：规划 **46/85** → 61%（略升，仍低于 CLS 63%）。

**S2 多 epoch 扫档**（正式协议 m_task，2026-09-23；同 run `20260920_083101`）：

| epoch | success | mean / median dist | mean mpc | 规划子集 |
|------:|--------:|-------------------:|---------:|----------|
| **3（planbest）** | **59%** | 3.39 / 3.32 | 5.83 | 44/85 |
| 10 | 26% | 4.18 / 3.81 | 7.76 | 11/85 |
| 20 | 47% | 3.35 / 3.21 | 5.92 | 32/85 |
| 30 | 35% | 3.35 / 3.44 | 7.06 | 20/85 |
| 40 | 38% | 3.29 / 3.46 | 6.68 | 23/85 |
| 45 | 29% | 3.42 / 3.30 | 7.52 | 14/85 |
| 50 | 33% | 4.09 / 4.01 | 7.26 | 18/85 |
| 59 | 57% | 3.58 / 3.51 | 6.00 | 42/85 |
| 60 | 51% | 3.36 / 3.13 | 6.09 | 36/85 |

汇总：`outputs/point_maze_s2_epoch_sweep_g25_t5i10_summary.txt`。曲线 U 形；主表固定 ep3，勿用 pred_loss best 冒充规划最优。

### 3.3 Rope — 早期架构 / cost 消融 / C1

| 标签 | 结果文件 | cost | n | success | mean / median Chamfer |
|------|----------|------|---|---------|------------------------|
| S2 早期（含 rel 等） | `rope_s2_best_metrics_n10.json` | m_task | 10 | 0% | 2.10 / 1.99 |
| **S2 norel cost 消融** | `rope_s2_norel_metrics_n10_dense.json` | **dense** | 10 | **10%** | **2.12 / 1.82** |
| C1（对照） | `rope_c1_metrics_n10.json` | m_task | 10 | 0% | 2.57 / 2.13 |

同权重 S2：m_task（主表 20% / 1.75）优于 dense 消融（10% / 2.12）。n=10 方差大。

### 3.4 Granular — 早期架构 / cost 消融

| 标签 | 结果文件 | cost | n | success | mean / median Chamfer |
|------|----------|------|---|---------|------------------------|
| S2 早期（非 norel） | `granular_s2_best_metrics_n10.json` | m_task | 10 | **100%** | 0.56 / 0.59 |
| **S2 norel cost 消融** | `granular_s2_norel_metrics_n10_dense.json` | **dense** | 10 | **100%** | **0.61 / 0.62** |

同权重 S2 norel：dense 消融成功率/Chamfer 好于 m_task 主表（80% / 0.76），但 **n=10 很噪**，仅作消融参考；主表仍报 m_task 以与 C1/S1 对齐。早期非 norel 勿与 norel 混报。

### 3.5 Cost 消融一句话（S2 norel，只改 `--cost-mode`）

| 环境 | m_task（主表） | dense（消融） |
|------|---------------|---------------|
| wall | **79%** | **73%**（变差） |
| point ep3 | **59%** | **61%**（略升，仍 < CLS 63%） |
| rope | **20%** / Chamfer 1.75 | **10%** / 2.12（变差） |
| granular | **80%** / 0.76 | **100%** / 0.61（变好，n=10） |

主表默认仍跟 C1 用 **`m_task_terminal`**。

### 3.6 S2 norel 末 epoch（正式协议，2026-09-26）

同标准 run、同正式时程 / cost，权重改为训练**最后一轮 epoch**（非主表 best / planbest）。勿与 §2 主表混列。

| 数据集 | ckpt | 结果文件 | success | mean / median |
|--------|------|----------|--------:|---------------|
| wall | `…_s2_20260919_235635/…_epoch_100_object.ckpt` | `outputs/wall_s2_results_n100_g25_t2i10_ep100.txt` | **76%** | 6.09 / 3.51（`state_dist`） |
| point | `…_s2_20260920_083101/…_epoch_60_object.ckpt` | `outputs/point_maze_s2_results_n100_g25_t5i10_ep60.txt` | **50%** | 3.44 / 3.37（`state_dist`） |
| rope | `…_s2_20260919_140440/…_epoch_100_object.ckpt` | `outputs/rope_s2_norel_metrics_n10_ep100.json` | **0%** | **2.22 / 2.07**（Chamfer） |
| granular | `…_s2_20260919_235635/…_epoch_100_object.ckpt` | `outputs/granular_s2_norel_metrics_n10_ep100.json` | **80%** | **0.77 / 0.79**（Chamfer） |

对照主表（当时 best/planbest）：wall 79%→76%；point ep3 59%→末 ep60 50%；rope 20%/1.75→0%/2.22；granular 80%/0.76→80%/0.77。

### 3.7 旧 S2 主表（100ep best / planbest；历史保留）

下列曾为 §2 主表 S2 列（正式协议、旧标准 ckpt）。**现行主表已换为 §2 的 150ep last**；此处仅对照，勿与新主表混报。

| 数据集 | 旧标准 ckpt | 结果文件 | success | mean / median |
|--------|-------------|----------|--------:|---------------|
| wall | `…_s2_20260919_235635/..._best_object.ckpt` | `outputs/wall_s2_results_n100_g25_t2i10.txt` | **79%** | 5.82 / 3.30（`state_dist`） |
| point | `…_s2_20260920_083101/..._best_ep3_object.ckpt` | `outputs/point_maze_s2_results_n100_g25_t5i10_ep3.txt` | **59%** | 3.39 / 3.32（`state_dist`） |
| rope | `…_s2_20260919_140440/..._best_object.ckpt` | `outputs/rope_s2_norel_metrics_n10.json` | **20%** | **1.75 / 1.89**（Chamfer） |
| granular | `…_s2_20260919_235635/..._best_object.ckpt` | `outputs/granular_s2_norel_metrics_n10.json` | **80%** | **0.76 / 0.62**（Chamfer） |

与现行 150ep last（§2）对照：wall 79%→78%；point ep3 59%→30%；rope 20%/1.75→30%/2.14（成功率升、Chamfer 变差）；granular 80%/0.76→80%/0.82。

### 3.8 S2 150ep best（正式协议，2026-09-28）

同 §2 的 **150ep 训**（`20260926/20260927` 四 run）、同正式时程 / cost，权重改为各训 **val `pred_loss` best**（`*_best_object.ckpt`）。勿与 §2 主表（150ep last）混列。

| 数据集 | best epoch | ckpt | 结果文件 | success | mean / median |
|--------|----------:|------|----------|--------:|---------------|
| wall | **147** | `…_s2_20260927_173022/..._best_object.ckpt` | `outputs/wall_s2_results_n100_g25_t2i10_ep150_best.txt` | **67%** | 7.06 / 3.54（`state_dist`） |
| point | **2** | `…_s2_20260927_173356/..._best_object.ckpt` | `outputs/point_maze_s2_results_n100_g25_t5i10_ep150_best.txt` | **37%** | 3.24 / 3.35（`state_dist`） |
| rope | **28** | `…_s2_20260926_142449/..._best_object.ckpt` | `outputs/rope_s2_norel_metrics_n10_ep150_best.json` | **0%** | **2.40 / 2.13**（Chamfer） |
| granular | **4** | `…_s2_20260926_161822/..._best_object.ckpt` | `outputs/granular_s2_norel_metrics_n10_ep150_best.json` | **80%** | **0.75 / 0.75**（Chamfer） |

对照同训 150ep last（§2）：wall 78%→best 67%；point 30%→37%；rope 30%/2.14→0%/2.40；granular 80%/0.82→80%/0.75。

### 3.9 CLS 末 epoch（正式协议，2026-09-29）

同标准 CLS run、同正式时程 / cost（`token=cls`，`dense_terminal`），权重改为训练**最后一轮**（`*_epoch_100_object.ckpt`，非主表 best）。勿与 §2 主表混列。

| 数据集 | ckpt | 结果文件 | success | mean / median |
|--------|------|----------|--------:|---------------|
| wall | `…_wall_v2_20260922_135024/…_epoch_100_object.ckpt` | `outputs/wall_cls_results_n100_g25_t2i10_ep100.txt` | **65%** | 7.04 / 3.65（`state_dist`） |
| point | `…_point_maze_v2_20260922_193111/…_epoch_100_object.ckpt` | `outputs/point_maze_cls_results_n100_g25_t5i10_ep100.txt` | **100%** | 2.97 / 2.81（`state_dist`） |
| rope | `…_rope_v3_20260922_154251/…_epoch_100_object.ckpt` | `outputs/rope_cls_metrics_n10_ep100.json` | **20%** | **2.15 / 2.65**（Chamfer） |
| granular | `…_granular_v1_20260922_193208/…_epoch_100_object.ckpt` | `outputs/granular_cls_metrics_n10_ep100.json` | **90%** | **0.64 / 0.62**（Chamfer） |

对照主表 CLS best：wall 70%→65%；point 63%→**100%**（early/`mpc_iters=0` 仍为 15/100）；rope 20%/1.48→20%/2.15（Chamfer 变差）；granular 100%/0.59→90%/0.64。

---

## 4. 配置背景（CLS / DINO / C1）

### 4.1 原版 CLS 训练要点

yaml 无 `wm.token_mode` / `plan` → 代码默认 `cls` + `dense_terminal`。  
表示 1×CLS；损失 dense MSE + SIGReg；wall/point fs=5，rope/granular fs=1。

### 4.2 与 DINO-WM 的关系

**不完全是 DINO。** 正式协议 = DINO 的任务时程 / 成功球 / CEM 内参 +（S1/S2）C1 的 `M_task` + 本仓库 MPC 截断（wall **2/10**、point **5/10**）。  
CLS 在时程上可对齐，但 **cost/表示都不是 DINO**。严格 DINO planner（`n_taken=5` + `max_iter=null` + visual cost）本仓库未接。

### 4.3 C1（对照方法，非本主表三列）

表示：`spatial16` + **avg pool**（无 query/global）；cost：`m_task_terminal`。  
评估时程与 S1/S2 主表相同。本机 wall/point/granular C1 结果文件多缺失；rope 有 `rope_c1_metrics_n10.json`（见 §3.3）。

### 4.4 原版 → C1 仅两处核心

1. 表示：`cls` → 16-token avg。  
2. 规划 cost：`dense` → `m_task`。  
训练仍是 dense MSE + SIGReg（不是 C2 训练 mask）。

---

## 5. 结果文件索引

### 5.1 正式主表

```
# wall
outputs/wall_cls_results_n100_g25_t2i10.txt
outputs/wall_s1_results_n100_g25_t2i10.txt
outputs/wall_s2_results_n100_g25_t2i10_ep150.txt          # S2 150ep last（现行）

# point
outputs/point_maze_cls_results_n100_g25_t5i10.txt
outputs/point_maze_s1_results_n100_g25_t5i10.txt
outputs/point_maze_s2_results_n100_g25_t5i10_ep150.txt    # S2 150ep last（现行）

# rope
outputs/rope_cls_metrics_n10.json
outputs/rope_s1_best_metrics_n10.json
outputs/rope_s2_norel_metrics_n10_ep150.json              # S2 150ep last（现行）

# granular
outputs/granular_cls_metrics_n10.json
outputs/granular_s1_best_metrics_n10.json
outputs/granular_s2_norel_metrics_n10_ep150.json          # S2 150ep last（现行）
```

### 5.2 消融 / 对照（节选）

```
outputs/wall_s2_results_n100_g25_t2i10_dense.txt
outputs/point_maze_s2_results_n100_g25_t5i10_ep3_dense.txt
outputs/point_maze_s2_results_n100_g25_t5i10_ep{10,20,30,40,45,50,59,60}.txt
outputs/point_maze_s2_epoch_sweep_g25_t5i10_summary.txt
outputs/point_maze_s2_results_n100_c1proto.txt          # ep3, g25, MPC 5/5
outputs/rope_s2_norel_metrics_n10_dense.json
outputs/granular_s2_norel_metrics_n10_dense.json
outputs/rope_c1_metrics_n10.json
outputs/s2_step2_cost_ablation_summary.txt
# 旧 S2 主表 best / planbest（§3.7）
outputs/wall_s2_results_n100_g25_t2i10.txt
outputs/point_maze_s2_results_n100_g25_t5i10_ep3.txt
outputs/rope_s2_norel_metrics_n10.json
outputs/granular_s2_norel_metrics_n10.json
# 旧 S2 末 epoch（正式协议，§3.6）
outputs/wall_s2_results_n100_g25_t2i10_ep100.txt
outputs/point_maze_s2_results_n100_g25_t5i10_ep60.txt
outputs/rope_s2_norel_metrics_n10_ep100.json
outputs/granular_s2_norel_metrics_n10_ep100.json
# S2 150ep best（正式协议，§3.8）
outputs/wall_s2_results_n100_g25_t2i10_ep150_best.txt
outputs/point_maze_s2_results_n100_g25_t5i10_ep150_best.txt
outputs/rope_s2_norel_metrics_n10_ep150_best.json
outputs/granular_s2_norel_metrics_n10_ep150_best.json
# CLS 末 epoch（正式协议，§3.9）
outputs/wall_cls_results_n100_g25_t2i10_ep100.txt
outputs/point_maze_cls_results_n100_g25_t5i10_ep100.txt
outputs/rope_cls_metrics_n10_ep100.json
outputs/granular_cls_metrics_n10_ep100.json
# 短程等：wall_*_t2i10.txt（无 g25）、wall_*_best_*、point_maze_s2_results_n100.txt 等
```
