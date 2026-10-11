# 仓库地图（2026-09-30，r3 默认）

## 一句话

**训练只看 `config/train/r3/`；评估用 `scripts/eval_r3.sh`；权重要去 `$STABLEWM_HOME/runs/r3/`。**

## 日常命令

```bash
export STABLEWM_HOME=/data/czx/le-wm/first/data
export CUDA_VISIBLE_DEVICES=0

# 训练（已烘焙 r3，无手动开关）
scripts/train_r3.sh rope s2
scripts/train_r3.sh wall cls
# 等价: .venv/bin/python train.py --config-name=r3/rope_s2

# 健康检查
.venv/bin/python scripts/repr_health.py --env rope \
  --ckpt data/runs/r3/rope_s2_s3072_.../rope_s2_last_object.ckpt

# 评估（方法 CLI 已按 cls/c1/s1/s2 写好）
scripts/eval_r3.sh rope s2 data/runs/r3/.../rope_s2_last_object.ckpt
```

| 方法 | 配置名后缀 | 含义 |
|------|-----------|------|
| CLS | `_cls` | 官方单 token |
| C1 | `_c1` | 16 avg-pool 槽 |
| S1 | `_s1` | query + 全局 + learned PE |
| S2 | `_s2` | S1 + NibblePE 逐层注入 |

## 目录约定

```text
config/train/
  r3/           ← 正式入口（环境×方法）
  recipe/       ← r3 / r2 配方
  method/       ← cls/c1/s1/s2 结构
  data/         ← hdf5 数据集定义
  _legacy/      ← 旧 lewm_*.yaml，勿用于新实验
  README.md

data/                 ← STABLEWM_HOME：*.h5 数据
  runs/r3/            ← 新训练权重与 run 内 config.yaml
  lewm_*              ← 旧 r1 run（历史，不混表）

outputs/r3/           ← 评估结果
scripts/              ← train_r3 / eval_r3 / 分析 / 健康检查
tools/                ← 转换、冒烟、旧评估脚本
_archive/             ← 日志备份等，不参与运行
docs/standard_r3.md   ← 现行标准
```

## 核心代码（留在根目录，Hydra/评估依赖）

`train.py` · `jepa.py` · `module.py` · `utils.py` · `eval_{wall,point_maze,rope_1,granular}*.py` · `setup_env.sh`

## 依赖目录（勿挪）

`AdaptiGraph/` · `mujoco210/` · `deformable_env/` · `conf/env/` · `env/`
