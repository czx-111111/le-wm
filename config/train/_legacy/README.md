# 旧训练配置（r1 / 早期）

这些 yaml 仍保留超参与注释，**不要**用于正式实验。

正式训练：

```bash
python train.py --config-name=r3/<env>_<method>
```

若必须复现旧 run，可：

```bash
python train.py --config-name=_legacy/lewm_rope_s2
```

注意：部分 `_s2` 配置 `defaults` 里写了相对名（如 `lewm_rope_s1`），挪到本目录后可能无法直接 compose；优先用当时 run 目录里保存的 `config.yaml`。
