# 训练配置目录说明

## 正式入口（只用这里）

```text
config/train/r3/<环境>_<方法>.yaml
```

| 环境 | 方法 |
|------|------|
| `wall` / `point_maze` / `rope` / `granular` | `cls` / `c1` / `s1` / `s2` |

```bash
export STABLEWM_HOME=/data/czx/le-wm/first/data
.venv/bin/python train.py --config-name=r3/wall_s2
# 或
scripts/train_r3.sh wall s2
```

无需再写 `+recipe=r3`，也无需手动设 batch / λ / SIGReg。

## 目录分层

| 路径 | 作用 |
|------|------|
| `r3/` | **你要训练的配置**（环境×方法，已烘焙 recipe=r3） |
| `recipe/` | 训练配方：`r3` 正式，`r2` 历史对照 |
| `method/` | 方法结构：cls / c1 / s1 / s2（只含模型开关） |
| `data/` | 数据集（hdf5 名、frameskip） |
| `_legacy/` | 旧版 `lewm_*.yaml`（r1 超参），勿用于新实验 |

## 权重落盘

`$STABLEWM_HOME/runs/r3/<env>_<method>_s<seed>_<时间戳>/`

内含 `config.yaml`、`*_last_object.ckpt`、`*_weights.ckpt` 等。
