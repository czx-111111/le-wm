# CAS-JEPA：基于官方 CLS 的全新顶刊方案

> **Controllability-Aware Spectral JEPA**  
> 在 **LeWM 官方单 token CLS** 上重启一条与 S1/S2/NibblePE/cost\* **完全正交** 的主线。  
> 目标：数学可陈述、四环境（wall / point_maze / rope / granular）可验证、冲顶会/顶刊（ICLR/ICML/NeurIPS 或 TMLR→扩展版）。  
> 状态：方法定稿（2026-10-11）。**不叠** 空间槽、NibblePE、`M_task`、`decoupled` cost\*。  
> **已删除**：逐步势能下降及一切「进度辅助损失」分支——主方法只保留可控谱几何。

---

## 0. 一句话与定位

**问题**：CLS-JEPA 用 \(\mathrm{MSE}+\lambda\cdot\mathrm{SIGReg}\) 学到各向同性高斯嵌入，但规划用 \(\|\hat z_H-z_g\|^2\)；各向同性把**不可控/视觉干扰方向**与**可控任务方向**同等对待，训练几何与规划代价不对齐。

**方法**：保持官方 CLS 架构不动，改**嵌入几何**：用动作雅可比刻画可控子空间，在该子空间上做谱 SIGReg，训练与规划共用度量 \(d_M\)。

**与旧 fork 的切割**：

| | S1/S2 线（已冻结，作对照） | **本线 CAS-JEPA（新）** |
|--|--|--|
| 表示 | 17 空间 token + NibblePE | **1× CLS（官方）** |
| 主故事 | tokenization / PE | **可控谱几何 + 训练–规划同构** |
| 规划 cost | `cost*` / `M_task` | **度量终端代价 \(d_M(\hat z_H,z_g)\)** |
| 失败教训 | C2/C3/S3 负结果 | 不复用那些开关 |

---

## 1. 顶刊叙事（投稿用）

### 1.1 标题候选

1. *Controllability-Aware Spectral Geometry for JEPA World Models*  
2. *When Isotropic Latents Hurt Planning: Spectral Metrics for JEPA*  
3. *Aligning Prediction and Planning Geometries in End-to-End JEPA*

### 1.2 核心 claim

> SIGReg 诱导的各向同性潜空间对**表示健康**有利，但对 **MPC/CEM 规划** 次优：任务相关变化往往落在低维可控子空间。我们把正则与预测限制在可控谱上，并用同一度量 \(d_M\) 做规划，使训练几何与规划代价同构。

### 1.3 贡献分层

1. **诊断（理论+实证）**：各向同性嵌入下，规划代价沿不可控方向的梯度噪声有显式上界；用 `repr_health` 的 ρ 与跨维方差谱，展示「高维方差 ≠ 可控任务方向」。  
2. **方法**：可控子空间上的谱 SIGReg + 共享度量预测/规划（§2）。**架构仍是 LeWM-CLS。**  
3. **实验协议**：沿用 r3（batch 128、健康检查、dev/test、配对检验、≥3/4 同向），相对 CLS /（可选）S2 作正交对照。

### 1.4 为何够「顶刊」

- **数学对象清晰**：可控 Gram / 投影、度量 \(M\succ 0\)，可证规划梯度 SNR 与代价 Lipschitz 界。  
- **直击 JEPA 规划缺口**：改「嵌入空间是什么几何」，不是再堆 heuristic loss。  
- **与 LeWM 互补**：LeWM 证明「两损失可稳定训练」；本文证明「稳定 ≠ 规划最优几何」。  
- **负空间干净**：不碰训练硬 mask / nibble CE；不加逐步势能约束（绕路任务有害且与度量预测冗余）。

---

## 2. 方法（公式定稿）

### 2.0 符号与骨干（冻结为官方 CLS）

- 编码：\(z_t = g_\theta(x_t)\in\mathbb{R}^d\)（ViT-tiny CLS → projector，\(d=192\)）。  
- 预测：\(\hat z_{t+1}=P_\phi(z_{\le t}, a_t)\)（AdaLN AR predictor）。  
- 基线损失（LeWM）：
\[
\mathcal{L}_{\mathrm{LeWM}}
=\underbrace{\|\hat z_{t+1}-z_{t+1}\|_2^2}_{\mathcal{L}_{\mathrm{pred}}}
+\lambda_{\mathrm{sig}}\,\mathrm{SIGReg}(z).
\]
- 规划（基线）：\(\min_{a_{0:H-1}}\;\|\hat z_H-z_g\|_2^2\)。

**本方法不改** encoder/predictor 拓扑、history=3、`num_preds`、r3 优化器与 λ 量级；只加可控几何模块与损失项。

### 2.1 可控谱分解（Controllable Spectral Factor）

对 batch 内样本，用有限差分估计动作灵敏度（训练图可截断）：

\[
J_t
=\frac{\partial P_\phi(z_t,a_t)}{\partial a_t}
\in\mathbb{R}^{d\times m}
\quad(m=2\text{ wall/point},\; m=4\text{ rope/granular}).
\]

经验可控 Gram：

\[
G
=\mathbb{E}_t\!\big[J_t J_t^\top\big]+\varepsilon I_d
\in\mathbb{S}_{++}^d.
\]

特征分解 \(G=U\Lambda U^\top\)，\(\Lambda=\mathrm{diag}(\lambda_1\ge\cdots\ge\lambda_d)\)。取能量阈值 \(\eta\in(0,1)\)（默认 0.95）定义可控秩 \(r\)：

\[
r
=\min\Big\{k:\sum_{i=1}^{k}\lambda_i\ge\eta\sum_{j=1}^{d}\lambda_j\Big\}.
\]

投影：

\[
\Pi_c=U_{:r}U_{:r}^\top,\qquad
\Pi_\perp=I-\Pi_c.
\]

实现：用 **滑动 EMA 累积 \(G\)**（随机 SVD / top-\(r\)，\(r\le 16\)），**stop-grad \(U\)** 进正则，避免雅可比–投影双层不稳。

**直觉**：wall/point 期望 \(r\) 小且稳；rope/granular 期望 \(r\) 更大但仍 \(\ll d\)。

### 2.2 度量与规划同构（Shared Metric）

默认构造 SPD 度量：

\[
M
= U_{:r}\,\mathrm{diag}(w)\,U_{:r}^\top
+\mu\,\Pi_\perp,
\qquad
w_i=\frac{\lambda_i}{\lambda_i+\tau}>0,\quad \mu\ll 1.
\]

即：**抬高可控方向权重，压低不可控方向对代价的贡献**。可选消融：对角 \(M=\mathrm{diag}(m)\)、低秩 \(M=I+LL^\top\)（\(k\le 8\)）。

\[
d_M(u,v)=(u-v)^\top M(u-v).
\]

**训练**：

\[
\mathcal{L}_{\mathrm{pred}}^{M}
=d_M(\hat z_{t+1},z_{t+1}).
\]

**规划（与训练同构）**：

\[
C(a_{0:H-1})
=d_M(\hat z_H,z_g).
\]

CLS 仍单 token，**无需** `M_task` / decoupled。

### 2.3 可控子空间上的谱 SIGReg（Spectral SIGReg）

\[
\mathcal{L}_{\mathrm{sig}}^{c}
=\mathrm{SIGReg}(U_{:r}^\top z)
+\beta\,\big\|\Pi_\perp z\big\|_2^2.
\]

- 第一项：只在可控坐标上做 sketch 高斯正则（复用现有 `SIGReg`）。  
- 第二项：不可控残差能量惩罚（防任务信息藏进 \(\Pi_\perp\)）。  
- \(\beta\) 建议 \(10^{-3}\)–\(10^{-2}\)。

### 2.4 总目标与理论草图

\[
\boxed{
\mathcal{L}_{\mathrm{CAS}}
=
\mathcal{L}_{\mathrm{pred}}^{M}
+\lambda_{\mathrm{sig}}\,\mathcal{L}_{\mathrm{sig}}^{c}
}
\]

规划：\(C=d_M(\hat z_H,z_g)\)。

| 符号 | 起点 | 说明 |
|------|------|------|
| \(\lambda_{\mathrm{sig}}\) | 0.09 | 与官方一致 |
| \(\eta\) | 0.95 | 可控能量 |
| \(\mu\) | \(10^{-2}\) | 不可控度量底 |
| \(\beta\) | \(5\times 10^{-3}\) | 残差惩罚 |
| \(r_{\max}\) | 16 | 谱截断上限 |

**命题（附录级）**：在线性化可控模型下，各向同性代价使规划梯度 SNR 按 \(r/d\) 衰减；投影到可控子空间并用 \(M\) 后 SNR 上界提升 \(d/r\)。模型误差以 Lipschitz 进入终端代价遗憾界。

### 2.5 与常见变体的差异

| 常见做法 | CAS |
|----------|-----|
| EMA teacher / BYOL | 否 |
| 多 token / 空间 PE | 否 |
| 像素重建 | 否 |
| 训练硬 mask | 否 |
| 仅改规划 cost、不改训练 | 否（必须同构） |
| 各向同性 SIGReg | **否** → 可控谱 SIGReg |
| 逐步势能 / 额外进度损失 | **否**（已删） |

---

## 3. 四数据集：假说与指标

沿用 [`standard_r3.md`](standard_r3.md) 协议；主指标不变。

| 环境 | 可控假说 | 主指标 | CAS 期望相对 CLS |
|------|----------|--------|------------------|
| **wall** | \(r\approx 2\sim4\)；门缝变化进 \(\Pi_c\) | success@2 | **↑** 或持平但 ρ↑ |
| **point_maze** | 迷宫进度低维；各向同性易被纹理干扰 | success@2 | **↑**（重点） |
| **rope** | \(m=4\) 但外观高维；各向同性伤害最大 | mean Chamfer↓ | **↓ Chamfer**（核心卖点） |
| **granular** | 类似 rope；CLS 已强 | mean Chamfer↓ | 不显著变差；力争 ↓ |

**跨环境门控**：≥3/4 同向 + 配对显著 + 超过训练 seed 噪声；rope 用 ρ≥0.25。

**论文图**：

1. 四环境 \(\lambda_i(G)\) 谱曲线 + 选定 \(r\)。  
2. `repr_health` ρ、跨帧 std；CAS ≥ CLS。  
3. 消融：去 \(\Pi_c\) / 仅改规划度量不改训练 / 各向同性 SIGReg。

---

## 4. 实现计划（代码落点）

### 4.1 最小侵入改动

| 文件 | 改动 |
|------|------|
| `module.py` | `ControllableMetric`（EMA Gram、投影、\(d_M\)）；`SIGReg` 输入改为 \(U^\top z\) |
| `jepa.py` | `get_cost` 增加 `metric_terminal`；可选缓存 \(M\) |
| `train.py` | `lejepa_forward`：度量 MSE、谱 SIGReg、雅可比/有限差分 |
| `utils.py` / Hydra | `method/cas.yaml` + `config/train/r3/{env}_cas.yaml` |
| `eval_*.py` | `--cost-mode metric_terminal` |
| `scripts/repr_health.py` | 可选：报 \(r\)、\(\|\Pi_c\Delta z\|/\|\Delta z\|\) |

### 4.2 配置骨架

```yaml
# config/train/method/cas.yaml
# @package _global_
wm:
  type: lewm
  token_mode: cls
  num_spatial_tokens: 1

loss:
  cas:
    enabled: true
    eta: 0.95
    r_max: 16
    mu: 0.01
    beta: 0.005
    jac_eps: 1.0e-3
    gram_ema: 0.99
  sigreg:
    weight: 0.09
    per_token: true

plan:
  cost_mode: metric_terminal
```

### 4.3 明确不做

1. 空间 16 token / NibblePE / query pool。  
2. C2/C3/S3 开关。  
3. 像素 Chamfer/坐标监督（保持 pure JEPA）。  
4. 逐步势能约束或额外进度排序损失。  
5. 未过健康检查的 ckpt 进主表；不在 test 上调参。

---

## 5. 实验路线图

GPU 约束同 r3。判定同 [`standard_r3.md`](standard_r3.md) §5。

| 阶段 | 内容 | 通过条件 | 估时 |
|------|------|----------|------|
| **A0** | 实现 CAS + 单测（\(M\succ 0\)、ckpt 含谱） | rope 2 epoch 不 NaN；健康可跑 | 1–2 天 |
| **A1** | **rope** × {CLS_r3, CAS} seed 3072 | 健康 OK；dev Chamfer 优于 CLS（bootstrap 不含 0） | ~2×1h + eval |
| **A2** | **wall** × {CLS, CAS} | success@2 不显著变差；\(r\) 合理 | 2×~3h |
| **A3** | **point_maze** × {CLS, CAS} | success@2 ↑ 或持平+ρ↑ | 2×~8h |
| **A4** | **granular** × {CLS, CAS} | Chamfer 不显著变差 | 2×~1h |
| **A5** | 消融：−Π_c / train≠plan / 各向同性 SIGReg | 证明谱度量与同构必要 | 短训 |
| **A6** | 噪声底 seed 1234；评估 seed 42/43 | 过 r3 门控 | — |
| **A7** | 三 seed + **test** 主表 | 论文表 | — |
| **A8**（可选） | vs r3 S2+cost\* 一行 | 正交轴对照 | — |

**门控**：A1 不过 → 调 \(\mu,\eta,\beta\)；**禁止**上空间 token 救火。

### 5.1 消融表

| ID | 设定 |
|----|------|
| B0 | 官方 CLS（MSE + 各向同性 SIGReg + dense cost） |
| B1 | \(d_M\) 只用于规划（训练仍 L2）→ 预期弱 |
| B2 | \(\mathcal{L}_{\mathrm{pred}}^{M}\) + 各向同性 SIGReg |
| B3 | 谱 SIGReg + 训练仍 L2 |
| B4 | **主方法 CAS** |
| B5 | train/plan 度量故意不同构 → 预期差 |

---

## 6. 论文结构草案

1. Intro：JEPA 规划的几何错配  
2. Preliminaries：LeWM / SIGReg / CEM  
3. Method：§2.1–2.4  
4. Theory sketch：SNR 与代价遗憾界  
5. Experiments：四环境 + 谱分析 + 消融 + 健康指标  
6. Related：JEPA、bisimulation / metric RL、DINO-WM  
7. Limit：线性化可控性是局部的；强随机变形体可扩展随机可控 Gram

**页预算**：方法 2.5 页、理论 1 页、实验 3 页。

---

## 7. 风险与备选

| 风险 | 缓解 |
|------|------|
| 雅可比贵 / 不稳 | 有限差分只抽 1 动作维/step；EMA \(G\)；stop-grad \(U\) |
| granular 已很强 | 主打 rope/point；granular 强调不伤 + 谱可解释 |
| 审稿人说像 bisimulation | 强调：无奖励、纯 JEPA、与 SIGReg 谱改写绑定、规划同构 |
| 与 S2 比输 | 主文打 CLS；S2 放正交轴附录 |

**备选**：若 A1 两周仍不过，改用轨迹差分 \(\Delta z\) 对动作回归学 \(J\)（线性探头），故事改为 *data-driven controllable basis*。

---

## 8. 成功标准

1. 叙述不依赖空间 token；训练度量 = 规划代价。  
2. r3 下 vs CLS：**≥3/4 同向**，且 rope 或 point 至少一处配对显著。  
3. 消融 B1/B5 显示同构必要。  
4. 健康检查全绿（rope 按环境门槛）。  
5. 有谱图与 ρ，不只涨点。

先投 **ICLR/ICML**；理论写满可冲 NeurIPS；扩写后可投 **TMLR / IJRR**。

---

## 9. 两周冲刺清单

- [ ] A0：代码 + `r3/rope_cas.yaml`  
- [ ] A1：rope CLS vs CAS（seed 3072）  
- [ ] 固定 \(\eta,\mu,\beta\)  
- [ ] A2–A4：wall / point / granular  
- [ ] A5：三条关键消融  
- [ ] 初稿 Method + 一页理论草图  
- [ ] A6–A7：多 seed + test（只跑一次）

---

## 10. 一句话总结

**CAS-JEPA = 官方 CLS + 可控谱度量上的预测与 SIGReg + 同构规划代价 \(d_M\)。**  
用数学改 JEPA 潜空间几何，不用更多 token；四数据集按 r3 门控打赢 CLS，即具备顶会/顶刊主线资格。
