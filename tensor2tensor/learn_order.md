我理解你想聚焦学习 Transformer 模型相关的代码。不过在动手删除之前，我得提醒你一个重要问题——**T2T 的代码耦合度很高，不能简单地"只留模型文件、删掉其

## 完成总结

从原来的 452 个 `.py` 文件精简到 49 个，删除了 403 个无关文件（其他模型、RL、可视化、部署、命令行工具、大部分测试和数据集生成器等），并清理了残留空目录和临时文件。所有删除都是基于从 `transformer.py` 自动追踪出的真实依赖闭包，保证了 transformer 代码的完整性，没有误删任何依赖。

保留下来的代码按学习重要性分成三层：

**核心主角**（先读这两个）
- `models/transformer.py` — Transformer 模型主体，编码器/解码器的组装
- `models/transformer_test.py` — 看模型怎么被调用、配什么参数的最佳参考

**模型的真正实现**（精读 `layers/`）
- `layers/common_attention.py` — 多头注意力、自注意力的核心实现，**这是 Transformer 的灵魂**
- `layers/transformer_layers.py` — 编码器层、解码器层、FFN 的实现
- `layers/common_layers.py`、`common_hparams.py`、`modalities.py` — 通用层、超参数基类、输入输出模态处理

**支撑框架**（按需查阅 `utils/`）
- `utils/t2t_model.py` — 模型基类（bottom→body→top→loss 四段式）
- `utils/registry.py` — 注册机制
- `utils/beam_search.py` — 解码时的束搜索
- `utils/optimize.py`、`learning_rate.py` — 优化器和学习率调度（Transformer 著名的 warmup 策略就在这）

## 建议的阅读顺序

先读 `transformer.py` 的整体结构（搭积木的逻辑）→ 深入 `common_attention.py` 理解注意力机制 → 看 `transformer_layers.py` 理解编码解码层 → 对照 `transformer_test.py` 理解怎么用 → 最后看 `learning_rate.py` 里的 warmup 调度。

如果删多了想恢复，因为这是 git 仓库，任何文件都能用 `git checkout -- <文件路径>` 找回来。

需要我现在带你逐行精读 `transformer.py` 或者 `common_attention.py`（注意力机制）吗？我可以用中文把关键逻辑讲清楚。