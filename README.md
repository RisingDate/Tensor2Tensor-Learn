# Tensor2Tensor-Learn

> Transformer 学习专用精简版仓库

本仓库基于 Google [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)（T2T）裁剪而来，**仅保留学习 Transformer 模型所需的核心代码**，去掉了其他模型、强化学习、可视化、部署、数据集生成器等无关内容，方便聚焦阅读和理解 Transformer 的工程实现。

## 代码结构

精简后保留的核心文件分为三层：

**核心主角**

- `tensor2tensor/models/transformer.py` —— Transformer 模型主体，编码器/解码器的组装
- `tensor2tensor/models/transformer_test.py` —— 模型如何被调用、如何配置参数的参考

**模型的真正实现（`layers/`）**

- `common_attention.py` —— 多头注意力、自注意力的核心实现，Transformer 的灵魂
- `transformer_layers.py` —— 编码器层、解码器层、FFN 的实现
- `common_layers.py`、`common_hparams.py`、`modalities.py` —— 通用层、超参数基类、输入输出模态处理

**支撑框架（`utils/`）**

- `t2t_model.py` —— 模型基类（bottom → body → top → loss 四段式）
- `registry.py` —— 注册机制
- `beam_search.py` —— 解码时的束搜索
- `optimize.py`、`learning_rate.py` —— 优化器与学习率调度（Transformer 经典的 warmup 策略）

## 建议的阅读顺序

先读 `transformer.py` 的整体结构（搭积木的逻辑）→ 深入 `common_attention.py` 理解注意力机制 → 看 `transformer_layers.py` 理解编码解码层 → 对照 `transformer_test.py` 理解用法 → 最后看 `learning_rate.py` 里的 warmup 调度。

详细的学习笔记见 `tensor2tensor/learn_order.md`。

## 致谢与许可

原始项目为 [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)，遵循 Apache 2.0 协议（见 `LICENSE`）。本仓库仅用于个人学习。
