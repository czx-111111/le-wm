# r3 训练入口

每个文件 = 一个环境 × 一种方法，已包含 recipe=r3。

```bash
python train.py --config-name=r3/rope_s2
python train.py --config-name=r3/wall_cls seed=1234
```

| 文件 | 方法 |
|------|------|
| `*_cls` | 官方单 token |
| `*_c1` | 16 avg-pool 槽 |
| `*_s1` | query + 全局 + learned PE |
| `*_s2` | S1 + NibblePE 逐层注入 |
