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

---

## `tensor2tensor/` 各子文件夹说明

### `models/` — 模型定义

| 文件 | 作用 |
|---|---|
| `transformer.py` | **核心主文件**。定义 `Transformer` 类及其所有变体（TransformerEncoder / TransformerScorer / TransformerMemory），包含编码、解码、训练和推理的完整逻辑，以及数十个预设超参数配置（`transformer_base`、`transformer_big` 等） |
| `transformer_test.py` | 单元测试。**理解模型怎么被调用的最佳参考**——展示了如何构造 features、传入超参数、断言输出 shape，相当于模型的"使用说明书" |
| `Learn.md` | 本项目的 Transformer 代码学习路径指南（即你现在看的这份文档） |
| `README.md` | 极简说明，列出该目录包含的模型 |

---

### `layers/` — 神经网络层实现（Transformer 的真正实现细节）

| 文件 | 作用 |
|---|---|
| `common_attention.py` | ⭐ **注意力机制的完整实现**，包含 `multihead_attention`（多头注意力核心）、`dot_product_attention`（缩放点积注意力）、各类 attention bias（padding mask、因果 mask、相对位置偏置）、位置编码（`add_timing_signal_1d`）、参数注意力等。这是整个代码库最重要的文件之一 |
| `transformer_layers.py` | Transformer 专用的可复用层：`transformer_prepare_encoder`（编码器输入预处理）、`transformer_encoder`（多层编码器堆叠）、`transformer_ffn_layer`（前馈网络，支持标准 dense/卷积/MoE 多种实现） |
| `common_layers.py` | 通用神经网络工具函数库：`layer_preprocess` / `layer_postprocess`（LayerNorm + 残差连接）、`dense_relu_dense`（两层全连接）、`shift_right_3d`（序列右移，teacher forcing 用）、dropout 封装、shape 操作等 |
| `common_hparams.py` | 定义超参数基类 `basic_params1()`，包含学习率、优化器、batch size、初始化方式等所有通用超参数的默认值。`transformer_base_v1()` 就是在此基础上扩展的 |
| `modalities.py` | 模态处理层（Modality）：负责模型输入/输出的转换。`SymbolModality` 处理文本的 token embedding 和 softmax 输出；`AudioModality` 处理音频特征；图像等其他模态也在此定义。相当于模型的"输入输出接口" |
| `transformer_memory.py` | Transformer-XL 风格的**循环记忆机制**：将序列分成多个块（chunk），每个块处理时可以 attend 到前一块的激活缓存，使模型能处理超长序列（突破固定上下文窗口限制） |
| `area_attention.py` | 面积注意力（Area Attention）：将相邻 token 聚合成矩形"面积"后再做注意力，适合语音等需要多尺度感知的任务 |
| `common_audio.py` | 音频处理工具：mel 频谱特征提取、音频数据增广等，供语音识别任务使用 |
| `common_image_attention.py` | 图像上的注意力机制：将图像像素或 patch 组织成序列，再做 Transformer 风格的注意力，用于图像生成/理解任务 |
| `common_video.py` | 视频处理工具：帧采样、时序特征处理等，供视频理解任务使用 |
| `discretization.py` | 向量离散化工具：向量量化（VQ）相关实现，用于 VQ-VAE 类模型或 VQ 门控的 MoE |
| `vq_discrete.py` | 向量量化离散化的具体实现，配合 `discretization.py` 使用，也是 VQ-MoE 门控机制的基础 |

---

### `utils/` — 训练与推理的支撑框架（按需查阅）

| 文件 | 作用 |
|---|---|
| `t2t_model.py` | **模型基类**（`T2TModel`）。定义了 T2T 的四段式 pipeline：`bottom`（模态底层，token→向量）→ `body`（模型主体，子类实现）→ `top`（模态顶层，向量→logits）→ `loss`（损失计算）。所有模型（包括 Transformer）都继承自此类 |
| `registry.py` | 注册表机制：`@registry.register_model`、`@registry.register_hparams` 等装饰器，允许通过字符串名字查找模型类和超参数，是 T2T 框架的"服务发现"系统 |
| `beam_search.py` | **束搜索算法**（Beam Search）：推理时在 TensorFlow 计算图中实现的束搜索，支持长度惩罚（`alpha` 参数）、EOS 检测、top-beams 返回。被 Transformer 推理流程直接调用 |
| `learning_rate.py` | 学习率调度策略：包括 Transformer 论文中著名的 **Noam warmup 策略**（先线性增大，再按 step^-0.5 衰减）、余弦衰减、常数等。对理解训练稳定性很重要 |
| `optimize.py` | 优化器封装：整合 Adam、AdamW、Adafactor（TPU 内存友好型）、MultistepAdam（梯度累积）等优化器的创建和梯度裁剪逻辑 |
| `adafactor.py` | Adafactor 优化器的完整实现：通过因式分解二阶矩估计来大幅节省内存（相比 Adam 节省约 50%），TPU 训练的首选优化器 |
| `expert_utils.py` | 混合专家（Mixture of Experts, MoE）工具：`PadRemover`（移除 padding 位置加速 FFN 计算）、`local_moe`/`local_moe_tpu`（局部 MoE 的路由和分发）、负载均衡损失计算 |
| `decoding.py` | 推理解码的高层封装：处理 beam search 结果的后处理、序列截断、多任务解码调度等 |
| `data_reader.py` | 数据读取：从 TFRecord 文件构建训练/评估数据集，处理 padding、batching、bucketing（按长度分组以减少 padding 浪费）等 |
| `metrics.py` | 评估指标：BLEU（机器翻译）、准确率、困惑度（perplexity）等的 TF 计算图实现 |
| `bleu_hook.py` | BLEU 分数的回调钩子（Hook），在训练评估时自动调用并记录翻译质量 |
| `hparam.py` / `hparams_lib.py` | 超参数对象的底层实现（`HParams` 类）：支持 `add_hparam`、从字符串解析、序列化等操作 |
| `mlperf_log.py` / `mlperf_tags.py` | MLPerf 基准测试日志工具，在关键操作处打印标准化日志，用于性能基准测试，不影响模型逻辑 |
| `scheduled_sampling.py` | 计划采样（Scheduled Sampling）：训练时以一定概率用模型自身预测结果代替真实目标（缩小训练/推理差距） |
| `multistep_optimizer.py` | 多步优化器：梯度累积 N 步后再更新参数，等效于扩大 N 倍的 batch size，用于显存受限时模拟大 batch 训练 |
| `quantization.py` | 模型量化工具：将权重从 float32 量化为低精度（如 int8），用于模型压缩和加速部署 |
| `rouge.py` / `sari_hook.py` | 文本摘要评估指标：ROUGE（召回率导向的重叠评估）和 SARI（文本简化专用指标） |
| `misc_utils.py` | 杂项工具函数：文件操作、日志、字符串处理等通用辅助函数 |
| `contrib.py` | TF contrib 模块的兼容封装，处理 TF 1.x 到 2.x 的 API 变化 |
| `yellowfin.py` | YellowFin 自适应优化器的实现（自动调节学习率和动量），较少使用 |

---

### `data_generators/` — 数据集与数据预处理

| 文件/目录 | 作用 |
|---|---|
| `problem.py` | **数据集基类**（`Problem`）：定义数据生成、特征规范、词表加载等接口，所有具体数据集都继承自此类 |
| `text_problems.py` | 文本任务的数据集基类：`Text2TextProblem`（序列到序列）、`Text2ClassProblem`（文本分类）等，定义了通用的文本处理 pipeline |
| `text_encoder.py` | 文本编码器：Subword（子词）分词、BPE、字符级编码等，将原始文本转换为 token ID 序列 |
| `tokenizer.py` | 分词器：基于空格和标点的基础分词，以及 WordPiece/SentencePiece 风格的子词切分 |
| `generator_utils.py` | 数据生成工具：TFRecord 文件写入、数据集 shuffle、样本序列化等通用数据处理函数 |
| `librispeech.py` | LibriSpeech 语音识别数据集的定义（被 `transformer.py` 直接 import，用于语音识别任务） |
| `speech_recognition.py` | 语音识别任务的通用数据处理基类 |
| `audio_encoder.py` | 音频编码器：原始音频波形到 mel 频谱特征的转换 |
| `multi_problem.py` | 多任务学习数据集：将多个 Problem 混合采样，用于预训练/微调的多任务设置 |
| `ops/` | C++ 自定义 TF Op：高性能的序列打包（pack sequences）和子词编码操作，作为 TF 算子注册 |
| `test_data/` | 用于单元测试的极小数据集（词表文件、语料文本） |

---

### `bin/` — 命令行入口脚本

| 脚本 | 作用 |
|---|---|
| `t2t-trainer` | **主训练脚本**：启动模型训练，通过命令行参数指定模型名、数据集、超参数等 |
| `t2t-decoder` | **推理解码脚本**：加载训练好的模型，对输入文本执行翻译/生成 |
| `t2t-datagen` | 数据生成脚本：将原始数据集预处理成 TFRecord 格式 |
| `t2t-eval` | 评估脚本：在验证集上计算模型指标（BLEU、准确率等） |
| `t2t-bleu` | 专用 BLEU 计算工具 |
| `t2t-exporter` | 模型导出脚本：将训练好的模型导出为 SavedModel 格式，用于部署 |
| `t2t-insights-server` | 启动注意力可视化服务器（配合 `insights/` 使用） |
| `t2t-query-server` | 启动在线推理服务器，接受 HTTP 请求并返回模型输出 |
| `t2t-translate-all` / `t2t-avg-all` | 批量翻译和模型集成（对多个 checkpoint 的输出取平均）工具 |
| `t2t-make-tf-configs` | 生成分布式训练所需的 TF_CONFIG 配置文件 |

---

### `insights/` — 注意力可视化工具

| 文件/目录 | 作用 |
|---|---|
| `insight_configuration.proto` | Protobuf 协议定义：规定可视化请求/响应的数据格式 |
| `polymer/` | **基于 Polymer（Web Components）的前端可视化界面**，包含：`attention_visualization/`（注意力权重热力图）、`graph_visualization/`（计算图可视化）、`explore_view/`（交互式探索视图）等组件 |

---

### `notebooks/` — Jupyter 教学笔记

| 文件 | 作用 |
|---|---|
| `hello_t2t.ipynb` | **入门必看**：T2T 框架的快速上手教程，演示训练和推理的完整流程 |
| `Transformer_translate.ipynb` | Transformer 机器翻译的端到端演示 |
| `asr_transformer.ipynb` | 语音识别（ASR）任务的 Transformer 应用示例 |
| `t2t_problem.ipynb` | 如何自定义新 Problem（数据集）的教程 |
| `hello_t2t-rl.ipynb` | 强化学习相关功能的演示（已删除大部分相关代码） |

---

### `visualization/` — 独立注意力可视化

| 文件 | 作用 |
|---|---|
| `attention.js` | 纯 JavaScript 实现的注意力权重可视化，可直接在浏览器中渲染注意力热力图，不依赖后端服务 |
| `TransformerVisualization.ipynb` | 在 Jupyter 中调用 `attention.js` 可视化 Transformer 注意力权重的示例 |

---

### `test_data/` — 测试数据与预训练 Checkpoint

| 文件/目录 | 作用 |
|---|---|
| `transformer_test_ckpt/` | 一个极小的预训练 Transformer checkpoint（1 步训练），供 `transformer_test.py` 在单元测试中加载验证推理流程 |
| `vocab.translate_ende_wmt*.subwords` | 英德翻译任务的子词词表文件（32k 和 8k 两种规模），测试时直接使用，无需重新生成 |
| `example_usr_dir/` | 用户自定义目录的示例（演示如何在 T2T 框架外添加自己的模型/数据集） |