# LeWM 空间化改进：S1 / S2 / S3 方法说明

本文档描述在 **LeWorldModel (LeWM)** 官方 CLS 基线之上，经 **C1** 验证后分三步推进的方法体系。  
核心目标：在**不爆炸序列长度**的前提下，用少量空间 token 保留细粒度几何，并让训练目标与规划代价对齐，从而**稳定超过 CLS**。

---

## 0. 问题与总体路线

### 0.1 观察到的现象（为何要改）

| 设定 | 经验结论 |
|------|----------|
| 原版 LeWM：1 个 CLS | 规划 baseline，全局读出能力强 |
| C1：16 token（4×4 平均池化）+ 评估 `M_task` | **只有评估加 mask 涨点**；纯 16 token 仍常输给 CLS |
| C2：训练再加 `M_dyn` mask | **变差**（训练 mask ≠ 评估 `M_task`，且与 dense MSE 叠在一起） |

诊断：

1. **固定 4×4 平均池化太钝**，小物体 / 门缝 / 绳子细节被抹掉；CLS 是学出来的全局注意力读出。
2. **预测变难**：1 个目标 → 16 个目标，容量与超参未变。
3. **训练 ≠ 规划**：训练 dense MSE，规划才用稀疏 `M_task`；强行把训练也改成硬 mask（C2）会伤世界模型 rollout。
4. **位置几乎无 2D 结构**：预测器里 16 个无结构可学习向量，细网格几何进不去。

### 0.2 总原则

- **基线冻结为 C1**：dense 训练 + 评估 `M_task`；**不跑、不叠 C2**。
- **分步实现、分步训练**：每步只多开一件事；代码可用开关一次合进仓库。
- **Hex / 十六进制不是主创新名字**：它是 **radix-16、与 16 槽对齐的两级离散空间码** 的写法；论文主贡献写「tokenization-aligned two-scale spatial code + 可学习读出 + 稀疏规划」。

### 0.3 三步一览

| 阶段 | 改什么 | 不改什么 | 配置示例 |
|------|--------|----------|----------|
| **C1（对照）** | 4×4 avg pool → 16 token；评估 `M_task` | 训练 dense MSE | `lewm_*_c1.yaml` |
| **S1** | CLS + 16 query 替换 avg pool | PE 仍为「加一次」learned；无辅助 CE | `lewm_*_s1.yaml` |
| **S2（当前代码）** | 共享 4×4 NibblePE + 每层注入 + 动态 `n_lo`；相对偏置关闭 | 读出冻结为 S1；无辅助 CE | `lewm_*_s2.yaml` |
| **S3（已移除）** | 曾在 S2 上加双 nibble CE | rope 上 0.05 / 0.005 都差于 S2，代码已退回 | — |

```
ViT (197 = 1 CLS + 196 patch)
        │
        ├─ C1:  drop CLS → 4×4 avg pool → 16 tokens
        │
        └─ S1+: keep CLS
                 + 16 queries cross-attn on patches → 16 slots
                 → 17 tokens (global + slots)
                        │
                        ├─ S1: learned PE（输入加一次）
                        └─ S2: 共享 4×4 NibblePE（动态 n_lo + 每层注入；无相对偏置）
                        │
                        ▼
              训练: dense MSE + λ·SIGReg
              规划: M_task on 16 slots；全局位固定参与 cost
```

---

## 1. 公共基础：C1 与规划协议

S1/S2 都建立在 C1 协议上，先写清公共部分。S3 已从代码移除，见 §5。

### 1.1 C1 表示

- Encoder：ViT-tiny，`patch_size=14`，`img_size=224` → 196 patch + 1 CLS。
- C1 读出：丢弃 CLS，对 196 patch 做 `adaptive_avg_pool2d` → **4×4 = 16** 语义 token。
- Predictor：时间 PE + 16 个无结构空间 PE，**只在输入加一次**；时间块因果注意力。
- 损失：\(\mathcal{L} = \mathrm{MSE}(\hat z, z) + \lambda_{\mathrm{sig}}\mathrm{SIGReg}(z)\)。
- **不加**训练时 `M_dyn`。

### 1.2 评估 / 规划：`M_task`

实现：`jepa.compute_m_task_mask`。

1. 当前帧 vs goal 的像素差（默认先反 ImageNet normalize 到 \([0,1]\) RGB）。
2. `adaptive_avg_pool2d` 到 `spatial_grid×spatial_grid`（默认 4×4）。
3. 二值化：`eps`（默认阈值 \(0.08\)）/ `topk` / `mean`。
4. 规划终端 cost：只对 mask=1 的槽做 mean；空 mask 回退全 1。

S1+ 若含全局 token：mask 变为 `\mathrm{cat}([1], M_{\mathrm{task}}^{16})`，**全局永远参与**，空间槽仍稀疏。

### 1.3 为何不把 C2 当主线

C2 训练 mask 是潜空间相邻帧 \(\|z_{t+1}-z_t\| > \tau\)（`M_dyn`），与规划用的「相对 goal 的像素差」`M_task` **不是同一件事**；且 dense `pred_loss` 仍在，只是额外加权。实验上变差，故 **S1/S2 不启用 `loss.m_dyn`**。

---

## 2. S1：可学习空间读出 + 全局 token

### 2.1 动机

平均池化把每个 4×4 粗格内的 patch **均匀平均**，无法像 CLS 那样盯住任务物体。  
S1 用 **可学习 query 对 patch 做 cross-attn**，并**保留 CLS 全局通道**，目标是先逼近或超过 CLS，再谈位置编码。

### 2.2 结构

实现：`jepa.SpatialQueryPool` + `JEPA.encode`（`pool_mode=query`）。

```
hidden = ViT last_hidden_state          # (BT, 197, H)
cls    = hidden[:, 0:1]                 # (BT, 1, H)
patches= hidden[:, 1:]                  # (BT, 196, H)

queries ∈ R^{1×16×H}                   # 4×4 网格 2D sincos 初始化 + 小噪声
attn   = softmax(q @ k^T / √H)         # 单头（为 S2 量化 n_lo 保留注意力图）
slots  = out_proj(attn @ v)             # (BT, 16, H)

tokens = cat([cls, slots], dim=1)       # (BT, 17, H)
emb    = Projector(tokens)              # → (B, T, 17, D)
```

同时缓存（S2 的 NibblePE 用，S1 训练不用）：

- `slot_attn`：`(B, T, 16, P)`
- `n_hi`, `n_lo`, `n_hi_content`：由注意力图量化得到的离散码（见 §3）

### 2.3 预测器与损失（与 C1 同协议）

- 序列长度 **N = 17**（下标 0 = 全局，1..16 = 槽）。
- 空间 PE：17 个 **无结构** `nn.Parameter`，**仅输入加一次**（`pos_mode=learned`，`inject_every_layer=false`）。
- 损失：17 个 token 上 dense MSE + SIGReg（空间维展平进 batch）。
- 规划：16 槽 `M_task` + 全局位恒为 1。

### 2.4 配置与训练

- 配置：`config/train/lewm_{wall,point_maze,granular,rope}_s1.yaml`
- 关键开关：
  - `wm.pool_mode: query`
  - `wm.include_global_token: true`
  - `wm.pos_mode: learned`
  - `wm.inject_every_layer: false`
  - `plan.cost_mode: m_task_terminal`
- 单卡对齐 C1 全局 batch：`devices: [0]`，`batch_size: 16`，`accumulate_grad_batches: 8` → **128**（对应 C1 的 `16×2卡×4`）。

```bash
export STABLEWM_HOME=/data/czx/le-wm/first/data
export CUDA_VISIBLE_DEVICES=0
python train.py --config-name=lewm_rope_s1   # 或 wall / point_maze / granular
```

评估（weights ckpt 重建时）：

```bash
--token-mode spatial16 --pool-mode query --include-global-token \
--cost-mode m_task_terminal
```

### 2.5 必须做的对照（S1 过关标准）

同一评估协议（`m_task_terminal`，空间设定；CLS 用 dense）：

1. 原版 **CLS**
2. **C1**：avg pool 16 + `M_task`
3. **S1**：query + CLS + `M_task`

**过关**：S1 明显优于 C1，并逼近或超过 CLS。  
过不了 → **不要上 S2**（细码 `n_lo` 会不可靠）。

---

## 3. 离散空间码（Nibble /「十六进制」在方法中的真实角色）

在进 S2 之前，先定义码本身。这是后续 PE 与辅助损失的共同语言。

### 3.1 一个字节 = 两个 nibble，与 16 槽对齐

不要把 hex 理解成「整图 16×16 行优先编号 `h=r*16+c` 再 `Embedding(256)`」——那与普通网格坐标同构，审稿人会判改名。

正确切分：

| 半字节 | 符号 | 含义 | 取值 |
|--------|------|------|------|
| 高 nibble | \(n_{\mathrm{hi}}\) | **哪一个槽**（4×4 槽身份） | \(0..15\)，槽 \((R,C)\) 满足 \(n_{\mathrm{hi}}=4R+C\) |
| 低 nibble | \(n_{\mathrm{lo}}\) | **槽内哪一细格**（该槽覆盖的 4×4） | \(0..15\)，\((r,c)\) 满足 \(n_{\mathrm{lo}}=4r+c\) |
| 完整字节 | \(h=(n_{\mathrm{hi}}\ll 4)\,\|\,n_{\mathrm{lo}}\) | 紧凑写法（0x00–0xFF） | 论文里叫 *byte / radix-16 code*，不必叫「十六进制位置编码」 |

整图细坐标：\((4R+r,\ 4C+c)\)，仍是 16×16，但**分级方式与模型切分一致**。

### 3.2 从注意力图量化码（内容相关）

实现：`jepa.nibble_codes_from_attn`。

1. 将每 query 的 `attn` \((BT,16,P)\) 看成空间图，双线性插值到 16×16。
2. 对第 \(i\) 个 query，取其对应粗槽窗口内的质量分布，求质心 → 量化为 \(n_{\mathrm{lo}}\)。
3. \(n_{\mathrm{hi}}\)：槽身份（query 下标 0..15）；另算全图质心落入的粗格 → \(n_{\mathrm{hi\_content}}\)。后者只在已放弃的 S3 里当过标签，当前训练不用。

要点：低 nibble **可随物体在槽内移动**，不是写死的格子中心。训练时需监控 \(n_{\mathrm{lo}}\) 的熵/覆盖，防止塌缩。

### 3.3 距离必须先解码

**禁止**对字节整数做 \(|h_i-h_j|\) 或 XOR 当 2D 距离（`0x0F` 与 `0x10` 整数相邻、网格上是换行）。  
相对量：拆 nibble → \((R,C)\)、\((r,c)\) → L1 / Chebyshev；跨槽时细差门控为 0。

---

## 4. S2：NibblePE（尺度对齐的两级位置编码）

### 4.1 动机

S1 已有可靠局部槽后，把「无结构 17 维 PE」换成与 tokenization 对齐的两级码，使预测器显式使用粗/细几何。  
**创新点写在这里的耦合**：radix-16 由 16 槽瓶颈决定；两级都是 4×4，可共享几何参数。

### 4.2 绝对位置

实现：`module.NibblePE`。

\[
e_{\mathrm{cell}}(u,v) = E_{\mathrm{row}}[u] + E_{\mathrm{col}}[v]
\quad (u,v\in\{0..3\})
\]

\[
\mathrm{PE}_{\mathrm{slot}} = \mathrm{MLP}_{\mathrm{coarse}}\big(e_{\mathrm{cell}}(R,C)\big)
  + \mathrm{MLP}_{\mathrm{fine}}\big(e_{\mathrm{cell}}(r,c)\big)
\]

- \(E_{\mathrm{row}}, E_{\mathrm{col}}\)：**两级共用**（归纳偏置：粗细同构）。
- 全局 token：单独可学习 `global_embed`。
- 可选消融：再加 `joint_embed[h]`（256 表）；主结果建议**不加**，避免故事退化成「一张查找表」。

### 4.3 相对位置偏置（注意力）

对 token \(i,j\)（仅槽–槽）：

- 粗差：\((\Delta R,\Delta C)\in\{-(G-1)..G-1\}^2\)，查表 \(B_{\mathrm{coarse}}\)
- 细差：仅当 \(n_{\mathrm{hi}}\) 相同才加 \(B_{\mathrm{fine}}(\Delta r,\Delta c)\)，否则门控 0

\[
\mathrm{bias}_{ij} = B_{\mathrm{coarse}}[\Delta R,\Delta C]
  + \mathbf{1}[\mathrm{same\ cell}]\cdot B_{\mathrm{fine}}[\Delta r,\Delta c]
\]

加到 block-causal mask 上（实现见 `ARPredictor.forward`）。

### 4.4 当前 S2 默认

原来的共享几何 NibblePE，只关掉相对偏置：

- `abs_mode: shared`：粗、细共用 `E_row + E_col`，再经 `MLP_coarse` / `MLP_fine`。
- `n_lo_mode: attn`：细码由注意力质心给出。
- `inject_every_layer: true`
- `use_relative_bias: false`：不加 `B_coarse` / `B_fine`。

按槽独立嵌入（`abs_mode: slot`）只作消融，不是当前默认。

### 4.5 配置

```yaml
wm:
  pos_mode: nibble
  inject_every_layer: true
  n_lo_mode: attn
  use_relative_bias: false
  abs_mode: shared
```

训练：读出仍是 query+CLS；**损失仍无 nibble CE**。

### 4.6 S2 消融表（证明「不是改名」）

**同一 S1 读出**，只换 PE：

| # | 设定 |
|---|------|
| 1 | S1：17 个无结构 learned PE |
| 2 | 2D RoPE / 正弦 |
| 3 | 16×16 因子化 `E_row[r]+E_col[c]`（\(r,c\in0..15\)） |
| 4 | 平坦 `Embedding(256)`（hex 当普通 ID） |
| 5 | NibblePE + 分级相对偏置（代码在，**不是当前默认**） |
| 6 | NibblePE，去掉参数共享（`abs_mode: slot`，消融） |
| 7 | **当前 S2**：NibblePE，去掉相对偏置 |

rope 上采用的是第 7 行，不是第 5 行。带相对偏置的版本没有更好；按槽独立嵌入更差。不要再把「分级相对偏置」写成当前 S2。

---

## 5. S3：已从代码移除（负结果）

当前仓库没有 S3。训练损失只有 dense MSE + SIGReg，没有 `nibble_hi_head` / `nibble_lo_head`，也没有 `lewm_*_s3.yaml`。

rope、同一 10 条 episode 上，双 nibble CE 两次都差于 §4.4 的 S2：

| | 成功率 | 平均 Chamfer |
|---|---|---|
| S2 | 20% | 1.75 |
| S3，权重 0.05 | 10% | 2.02 |
| S3，权重 0.005 | 10% | 2.37 |

标签是注意力质心再量化，和规划用的 `M_task` 不是同一件事。权重大时分类头把嵌入拉去服务离散码；权重小时头学不会，噪声仍反传进预测器。主方法停在 S2，不要把相对偏置或分类损失加回来。

---

## 6. 实验协议建议

### 6.1 环境

与 LeWM / 现有脚本对齐：wall、point_maze、rope、granular（及已有评估脚本）。

### 6.2 训练配方（S1 默认）

| 项 | 值 |
|----|-----|
| Epochs | 100 |
| 优化器 | AdamW（各环境 lr / wd 见对应 yaml） |
| 精度 | bf16-mixed |
| 全局 batch | 128（单卡：16×8 accum） |
| SIGReg weight | 见各 yaml（rope 常 0.09，其它常 0.05） |
| History / num_preds | 3 / 1 |

### 6.3 规划评估

- Cost：`m_task_terminal`（阈值等与 C1 一致，便于公平对比）。
- 报告：成功率 / 距离指标，多 seed；**S1 必须同时报对 C1 与对 CLS**。

### 6.4 论文叙事（建议）

**弱标题**：「我们提出十六进制位置编码」  

**强问题**：

> JEPA 世界模型从 CLS 换成少空间 token 后，固定网格平均会丢掉任务物体，规划信号被背景稀释；如何在序列不爆炸的前提下，用可学习局部槽 + 与 tokenization 对齐的两级离散空间码 + 稀疏规划 cost，超过 CLS？

贡献分层：

1. **诊断与对齐**：评估 `M_task` 有效、训练硬 mask（C2）有害。  
2. **S1 读出**：全局 + 可学习槽。  
3. **S2 NibblePE**：radix-16 两级码、共享 4×4 几何、动态 `n_lo`、每层注入；相对偏置关闭（hex 是码的紧凑表示）。  
4. **S3**：rope 上双 nibble CE 为负结果，已从代码移除。

---

## 7. 代码与配置索引

| 内容 | 路径 |
|------|------|
| Query 读出 / nibble 量化 / M_task | `jepa.py`（`SpatialQueryPool`, `nibble_codes_from_attn`, `compute_m_task_mask`） |
| NibblePE / 预测器 | `module.py`（`NibblePE`, `ARPredictor`） |
| 工厂 | `utils.py`（`build_lewm`） |
| 损失 | `train.py`（`lejepa_forward`：dense MSE + SIGReg） |
| C1 配置 | `config/train/lewm_*_c1.yaml` |
| S1 配置 | `config/train/lewm_*_s1.yaml` |
| S2 | `config/train/lewm_{wall,rope,granular,point_maze}_s2.yaml` |
| 数据 | `config/train/data/{wall,point_maze,rope,granular}.yaml`；`STABLEWM_HOME/.../datasets/*_train.h5` |

关键开关：

| 键 | C1 | S1 | S2 |
|----|----|----|-----|
| `wm.pool_mode` | `avg` | `query` | `query` |
| `wm.include_global_token` | false | true | true |
| `wm.pos_mode` | learned | learned | **nibble** |
| `wm.inject_every_layer` | false | false | **true** |
| `wm.n_lo_mode` | — | — | **attn** |
| `wm.use_relative_bias` | — | — | **false** |
| `wm.abs_mode` | — | — | **shared** |
| `loss.m_dyn` | 无 | 无 | 无 |
| `plan.cost_mode` | `m_task_terminal` | 同左 | 同左 |

---

## 8. 明确不要做的事

1. 把 16 语义 token 扩成 256。  
2. 用 `Embedding(256)` 或「hex 更省 token / 更能自主学习」当主创新。  
3. 对字节整数做距离 / RoPE。  
4. 在 S1 同时换读出 + 每层 PE + 辅助 CE。  
5. 为「训练评估一致」把世界模型改成只预测 mask 内格子（rollout 会崩）。  
6. 继续加大 C2 的 `m_dyn.weight` 当主攻。  
7. 污染已有 `*_c1.yaml`；新实验用 `*_s1/s2`。不要把已删除的 nibble CE 加回来。

---

## 9. 推荐执行顺序（检查清单）

- [ ] 固定 C1 数字（各环境 + `M_task`）与 CLS 数字  
- [ ] 训 S1（建议先 rope / wall），对照 C1 与 CLS  
- [ ] S1 过关后再训 S2；做 PE 消融表 §4.6  
- [x] S3（双 nibble CE）在 rope 上为负结果，代码已退回 S2  
- [ ] 多环境复现；写清负结果（C2、相对偏置、S3）

---

## 10. 一句话总结

- **S1**：用 **CLS + 16 个可学习 query** 替换钝的平均池化，解决「空间 token 输给 CLS」的第一瓶颈。  
- **S2（当前代码）**：用 **共享 4×4 的两级 NibblePE**，细码跟着注意力质心走，并且每层注入；相对偏置关闭。「十六进制」只是该码的紧凑写法。  
- **S3**：双 nibble CE 在 rope 上不如 S2，已从代码移除。  

三者叠加的主线是：**少 token 的空间 JEPA + 任务稀疏规划 + tokenization-aligned 离散空间码**，而不是单独卖一种 PE 名字。
