# 标准训练 run / checkpoint 登记

> 更新时间：2026-09-28  
> 用途：标明**正式对比表**应使用的训练 run 与权重；避免与 aborted、消融、短程评测、非 norel 旧 S2 混用。  
> 评估协议见 [`S1_S2_eval_protocol.md`](S1_S2_eval_protocol.md)。  
> 时间均为 CST；`开跑` = run 目录名中的时间戳；`best 落盘` = 该 ckpt 文件 mtime（`*_best_ep*` 常为事后导出，≠ 该 epoch 训完时刻）。

---

## 0. 怎么算「标准训练」

同时满足：

1. **配置**：对应阶段的 yaml（CLS 无后缀；C1/S1/S2 用 `*_c1/_s1/_s2`）。  
2. **配方**：全局 batch **128**（常见单卡 `batch16 × accum8`，或双卡 `16×4×2`）；`seed=3072`。  
3. **方法**：S2 仅认 **norel**（`pos_mode=nibble`，`inject_every_layer`，`use_relative_bias=false`）。  
4. **权重**：  
   - CLS / C1 / S1：下列表中的 `*_best_object.ckpt`（point S1 另有标注的 epoch 档）。  
   - **S2（150 epoch 新标准）**：仍保存 `*_best_object.ckpt`，但**正式评估用最后一轮**（`*_last_object.ckpt` ≡ `*_epoch_150_object.ckpt`）。  
5. **评估**：必须再按 [`S1_S2_eval_protocol.md`](S1_S2_eval_protocol.md) 正式协议评（wall/point g25 + 正式 MPC；rope/granular 开环 g5；CLS=`dense`，C1/S1/S2=`m_task`）。

**不算标准**：`*_aborted_*`、wandb hijack 残次、goal=5 短程专用权重混报、带 relative bias / 无 inject 的早期 S2、未登记的平行 run、S2 未跑满 150 的中断目录。

---

## 1. 总表（正式用）

| 数据集 | 方法 | 标准？ | 训练配置 | run（开跑） | 选用 ckpt | best / last 落盘 |
|--------|------|--------|----------|-------------|-----------|------------------|
| wall | **CLS** | ✅ | `lewm_wall.yaml` | `lewm_wall_v2_20260922_135024`（09-22 13:50） | `…/lewm_wall_v2_best_object.ckpt` | best 09-22 23:31 |
| wall | **S1** | ✅ | `lewm_wall_s1.yaml` | `lewm_wall_s1_20260917_104917`（09-17 10:49） | `…/lewm_wall_s1_best_object.ckpt` | best 09-17 20:30 |
| wall | **S2** | ✅ **150ep** | `lewm_wall_s2.yaml`（norel） | **`lewm_wall_s2_20260927_173022`**（09-27 17:30） | **`…/lewm_wall_s2_last_object.ckpt`**（= epoch_150） | last 09-28 02:41；best 09-28 02:32 |
| wall | C1 | ❌ 本机无正式 run | `lewm_wall_c1.yaml` | — | — | — |
| point | **CLS** | ✅ | `lewm_point_maze.yaml` | `lewm_point_maze_v2_20260922_193111`（09-22 19:31） | `…/lewm_point_maze_v2_best_object.ckpt` | best 09-22 19:54 |
| point | **S1** | ✅ | `lewm_point_maze_s1.yaml` | `lewm_point_maze_s1_20260920_193429`（09-20 19:34） | **`…/lewm_point_maze_s1_best_ep8_object.ckpt`**（正式评测用） | 导出 09-22 13:34 |
| point | **S2** | ✅ **150ep** | `lewm_point_maze_s2.yaml`（norel） | **`lewm_point_maze_s2_20260927_173356`**（09-27 17:33；中断后 09-28 19:37 自 ep144 resume） | **`…/lewm_point_maze_s2_last_object.ckpt`**（= epoch_150） | last 09-28 20:14；best 09-27 18:38 |
| point | C1 | ❌ 本机无权重文件 | `lewm_point_maze_c1.yaml` | — | — | — |
| rope | **CLS** | ✅ | `lewm_rope.yaml` | `lewm_rope_v3_20260922_154251`（09-22 15:42） | `…/lewm_rope_v3_best_object.ckpt` | best 09-22 17:41 |
| rope | **C1** | ✅ | `lewm_rope_c1.yaml` | `lewm_rope_c1_20260922_231708`（09-22 23:17） | `…/lewm_rope_c1_best_object.ckpt` | best 09-22 23:36 |
| rope | **S1** | ✅ | `lewm_rope_s1.yaml` | `lewm_rope_s1_20260916_201520`（09-16 20:15） | `…/lewm_rope_s1_best_object.ckpt` | best 09-16 21:15 |
| rope | **S2** | ✅ **150ep** | `lewm_rope_s2.yaml`（norel） | **`lewm_rope_s2_20260926_142449`**（09-26 14:24） | **`…/lewm_rope_s2_last_object.ckpt`**（= epoch_150） | last 09-26 16:17；best 09-26 14:49 |
| granular | **CLS** | ✅ | `lewm_granular.yaml` | `lewm_granular_v1_20260922_193208`（09-22 19:32） | `…/lewm_granular_v1_best_object.ckpt` | best 09-22 22:12 |
| granular | **S1** | ✅ | `lewm_granular_s1.yaml` | `lewm_granular_s1_20260917_083958`（09-17 08:39） | `…/lewm_granular_s1_best_object.ckpt` | best 09-17 08:45 |
| granular | **S2** | ✅ **150ep** | `lewm_granular_s2.yaml`（norel） | **`lewm_granular_s2_20260926_161822`**（09-26 16:18） | **`…/lewm_granular_s2_last_object.ckpt`**（= epoch_150） | last 09-26 17:56；best 09-26 16:21 |
| granular | C1 | ❌ 本机无正式 run | `lewm_granular_c1.yaml` | — | — | — |

路径前缀均为 `data/<run_id>/`。

### 1.1 S2 150 epoch 配方（四数据集共用）

| 项 | 值 |
|----|-----|
| `max_epochs` | **150**（相对原 S1/旧 S2 的 100 ×1.5） |
| `save_object_every_n_epochs` | **10**（另存 best；结束再存 `*_last_object.ckpt`） |
| norel | `pos_mode=nibble`，`inject_every_layer=true`，`use_relative_bias=false`，`abs_mode=shared` |
| 正式评测权重 | **`*_last_object.ckpt`**（不要用 best 当主表，除非做消融） |

---

## 2. 旧 point S2（100ep 量级，历史主表）

> 以下为 **09-20** 旧标准训（约 60 epoch 收束），主表曾报 **ep3**。新正式 S2 请改用 §1 的 **150ep last**；本节仅作历史对照。

run：`lewm_point_maze_s2_20260920_083101`（开跑 **09-20 08:31**；09-21 曾从 ep56 resume 至约 60）。

| 档位 | ckpt | 说明 | 正式协议结果（g25，MPC 5/10，n=100） |
|------|------|------|--------------------------------------|
| **ep3** | `lewm_point_maze_s2_best_ep3_object.ckpt` | 全局 val 最优档；权重约 **09-20 13:39** 成为 best；文件 mtime 09-21 18:56 为导出 | **59%**（`outputs/point_maze_s2_results_n100_g25_t5i10_ep3.txt`） |
| ep45 | `…_best_ep45_object.ckpt` | 第二段 best | 29% |
| ep59 | `…_best_ep59_object.ckpt` | 第三段 best | 57% |

历史 `point_maze_s2_results_n100_c1proto.txt`（ep3，**MPC 5/5**）**不是**现行正式协议。

---

## 3. 完整路径速查

```
# wall
data/lewm_wall_v2_20260922_135024/lewm_wall_v2_best_object.ckpt
data/lewm_wall_s1_20260917_104917/lewm_wall_s1_best_object.ckpt
data/lewm_wall_s2_20260927_173022/lewm_wall_s2_last_object.ckpt          # S2 150ep 正式
data/lewm_wall_s2_20260927_173022/lewm_wall_s2_best_object.ckpt          # 同训 best（对照）
data/lewm_wall_s2_20260919_235635/lewm_wall_s2_best_object.ckpt          # 旧 S2（100ep 级）

# point
data/lewm_point_maze_v2_20260922_193111/lewm_point_maze_v2_best_object.ckpt
data/lewm_point_maze_s1_20260920_193429/lewm_point_maze_s1_best_ep8_object.ckpt
data/lewm_point_maze_s2_20260927_173356/lewm_point_maze_s2_last_object.ckpt   # S2 150ep 正式
data/lewm_point_maze_s2_20260927_173356/lewm_point_maze_s2_best_object.ckpt   # 同训 best（对照）
data/lewm_point_maze_s2_20260920_083101/lewm_point_maze_s2_best_ep3_object.ckpt   # 旧
data/lewm_point_maze_s2_20260920_083101/lewm_point_maze_s2_best_ep45_object.ckpt
data/lewm_point_maze_s2_20260920_083101/lewm_point_maze_s2_best_ep59_object.ckpt

# rope
data/lewm_rope_v3_20260922_154251/lewm_rope_v3_best_object.ckpt
data/lewm_rope_c1_20260922_231708/lewm_rope_c1_best_object.ckpt
data/lewm_rope_s1_20260916_201520/lewm_rope_s1_best_object.ckpt
data/lewm_rope_s2_20260926_142449/lewm_rope_s2_last_object.ckpt           # S2 150ep 正式
data/lewm_rope_s2_20260926_142449/lewm_rope_s2_best_object.ckpt           # 同训 best（对照）
data/lewm_rope_s2_20260919_140440/lewm_rope_s2_best_object.ckpt           # 旧 S2

# granular
data/lewm_granular_v1_20260922_193208/lewm_granular_v1_best_object.ckpt
data/lewm_granular_s1_20260917_083958/lewm_granular_s1_best_object.ckpt
data/lewm_granular_s2_20260926_161822/lewm_granular_s2_last_object.ckpt   # S2 150ep 正式
data/lewm_granular_s2_20260926_161822/lewm_granular_s2_best_object.ckpt   # 同训 best（对照）
data/lewm_granular_s2_20260919_235635/lewm_granular_s2_best_object.ckpt   # 旧 S2
```

---

## 4. 易混、不要当标准主表

| 路径 / run | 原因 |
|------------|------|
| `data/lewm_*_*_aborted_*` | 中断 / wandb 劫持，已废弃 |
| `lewm_rope_s2_20260919_140440` 等 **旧 100ep 级** S2 | 已被 §1 的 **150ep last** 取代；可作历史对照 |
| `lewm_rope_s2_20260919_162111`、`20260926_142319` 等其它 rope S2 | 非正式 150ep 登记 run |
| `lewm_granular_s2_20260918_072958` | 早期非 norel / 旧评测 |
| `lewm_wall_s2_20260926_175701` | 150ep 开跑后中断（无 epoch_150）；正式是 **`20260927_173022`** |
| `lewm_point_maze_s2_20260927_173022` | 误开 90ep 后立刻杀掉的残次目录 |
| S2 `*_best_object.ckpt`（150ep 新训） | **仍保留**，但主表评测请用 **`*_last_object.ckpt`** |
| 旧 point S2 `best_object.ckpt`（未带 ep 后缀） | 指向会变；旧主表请写明 ep3/45/59 |
| wall/point C1 | 配置有、本机无已登记标准权重 |

---

## 5. 训练结束时间（有日志可核的）

| run | 日志依据的大致结束 |
|-----|-------------------|
| wall CLS | 09-22 23:31（`lewm_wall_cls_resume_train.log` EXIT:0） |
| point CLS | 09-23 06:24 |
| rope CLS | resume 后约 09-22 19:45 |
| rope C1 | 09-23 01:45 |
| granular CLS | resume 后约 09-22 22:20 |
| point S2（旧） | resume 段 09-21 傍晚结束（EXIT:0） |
| **rope S2 150ep** | 09-26 16:17（`lewm_rope_s2_ep150_train.log` EXIT:0） |
| **granular S2 150ep** | 09-26 17:56（`lewm_granular_s2_ep150_train.log` EXIT:0） |
| **wall S2 150ep** | 09-28 02:41（`lewm_wall_s2_ep150_train.log` EXIT:0） |
| **point S2 150ep** | 09-28 20:14（`lewm_point_maze_s2_ep150_resume_train.log` EXIT:0；自 ep144 resume 收尾） |

其余 S1 / 旧 S2 以开跑 + best 落盘为准即可。
