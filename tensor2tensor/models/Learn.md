# Transformer 代码学习路径指南

本文件帮助你按照最合理的顺序阅读和理解 `transformer.py` 及其依赖模块的代码。
建议先通读论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 再对照代码学习。

---

## 整体代码地图

```
tensor2tensor/
├── layers/
│   ├── common_hparams.py       # 超参数基础定义
│   ├── common_layers.py        # 通用层（LayerNorm、dropout、dense等）
│   ├── common_attention.py     # 注意力机制核心实现
│   └── transformer_layers.py  # Transformer 专用层（编码器、FFN）
└── models/
    └── transformer.py          # Transformer 模型主体
```

---

## 第一阶段：超参数与基础工具（地基）

> 目标：了解模型的"配置表"和底层工具函数，为后续阅读做铺垫。

### Step 1 — `transformer_base_v1()` · `transformer_base()`
**文件**：`transformer.py` 第 2371 行起

这是最好的起点。在读任何计算逻辑之前，先弄清楚"模型有哪些旋钮"：

| 超参数 | 含义 |
|---|---|
| `hidden_size=512` | 每个 token 的向量维度（d_model） |
| `num_hidden_layers=6` | 编码器和解码器各 6 层 |
| `num_heads=8` | 多头注意力的头数 |
| `filter_size=2048` | FFN 中间层维度（通常是 hidden_size 的 4 倍） |
| `label_smoothing=0.1` | 标签平滑系数 |
| `layer_preprocess_sequence` | 控制 pre-norm 还是 post-norm |

读完后再看 `transformer_big()`、`transformer_tiny()`、`transformer_test()` 对比差异，感受各参数的意义。

---

### Step 2 — `common_hparams.basic_params1()`
**文件**：`layers/common_hparams.py`

了解 hparams 对象的基础字段（学习率、优化器、batch_size 等），
这些字段被 `transformer_base_v1()` 继承和覆盖。

---

## 第二阶段：注意力机制（核心原理）

> 目标：理解 Transformer 最核心的计算——多头注意力。

### Step 3 — `attention_bias_ignore_padding()` · `attention_bias_lower_triangle()`
**文件**：`layers/common_attention.py`

学习 attention bias（注意力偏置）的概念：
- `attention_bias_ignore_padding`：给 padding 位置施加 -1e9，让注意力忽略它们
- `attention_bias_lower_triangle`：下三角矩阵 mask，实现因果注意力（解码器专用）

这两个函数是后续所有注意力调用的前提知识。

---

### Step 4 — `multihead_attention()`
**文件**：`layers/common_attention.py`

这是整个代码库最核心的函数。理解它等于理解了论文第 3.2 节。

关键流程：
```
输入 query, memory, bias
    ↓
线性投影 → Q, K, V（各 num_heads 份）
    ↓
split_heads：拆分多头 → [batch, heads, length, depth_per_head]
    ↓
dot_product_attention：QK^T / sqrt(d_k) + bias → softmax → 加权 V
    ↓
combine_heads：合并多头
    ↓
线性输出投影
```

重点关注：
- `memory=None` 时是**自注意力**（Q=K=V）
- `memory≠None` 时是**交叉注意力**（Q 来自解码器，K/V 来自编码器）
- `cache` 参数：快速解码时缓存历史 K/V，避免重复计算

---

### Step 5 — `add_timing_signal_1d()`
**文件**：`layers/common_attention.py`

正弦/余弦位置编码的实现，对应论文第 3.5 节：
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## 第三阶段：编码器（Encoder）

> 目标：理解编码器如何把输入序列变成上下文表示。

### Step 6 — `transformer_prepare_encoder()`
**文件**：`layers/transformer_layers.py` 第 71 行

编码器的"准备工作"，按顺序做：
1. 计算 `encoder_self_attention_bias`（屏蔽 padding）
2. 加入目标空间嵌入（target_space_embedding，多任务/多语言时使用）
3. 加入位置编码

**输入**：词嵌入矩阵 `[batch, length, hidden_dim]`  
**输出**：`(encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias)`

---

### Step 7 — `transformer_ffn_layer()`
**文件**：`layers/transformer_layers.py` 第 452 行

前馈网络（FFN）的实现，对应论文第 3.3 节：
```
FFN(x) = max(0, x·W1 + b1)·W2 + b2
```

默认使用 `dense_relu_dense`（两层全连接 + ReLU）。
注意 `pad_remover` 优化：训练时跳过 padding 位置的 FFN 计算，节省算力。

---

### Step 8 — `transformer_encoder()`
**文件**：`layers/transformer_layers.py` 第 272 行

多层编码器堆叠：
```python
for layer in range(num_encoder_layers):
    x = self_attention(x) + x       # 子层1：自注意力 + 残差
    x = ffn(x) + x                  # 子层2：FFN + 残差
return layer_norm(x)                 # 最终 LayerNorm
```

注意 `layer_preprocess` / `layer_postprocess` 控制 pre-norm 还是 post-norm（v1 vs v2 的差异）。

---

### Step 9 — `transformer_encode()`
**文件**：`transformer.py` 第 89 行

编码器的顶层封装函数，组合了 Step 6-8：
1. 展平 4D → 3D
2. 调用 `prepare_encoder_fn` 准备输入
3. 施加 dropout
4. 调用 `encoder_function` 执行多层编码
5. 返回 `(encoder_output, encoder_decoder_attention_bias)`

---

## 第四阶段：解码器（Decoder）

> 目标：理解解码器如何利用编码器输出自回归地生成序列。

### Step 10 — `transformer_prepare_decoder()`
**文件**：`transformer.py` 第 1851 行

解码器的"准备工作"：
1. 计算因果 mask（`attention_bias_lower_triangle`）
2. 将目标序列**右移一位**（teacher forcing）：`[A, B, C]` → `[<GO>, A, B]`
3. 加入位置编码

**为什么右移**：预测第 t 个 token 时，解码器输入是第 t-1 个真实 token。

---

### Step 11 — `transformer_self_attention_layer()`
**文件**：`transformer.py` 第 1941 行

单个解码器的注意力部分，包含：
1. **自注意力**：query/key/value 均来自解码器，配合因果 mask
2. **交叉注意力**：query 来自解码器，key/value 来自编码器输出

快速解码时通过 `cache` 缓存历史 K/V，只计算当前新 token 的部分。

---

### Step 12 — `transformer_decoder_layer()`
**文件**：`transformer.py` 第 2111 行

完整的单个解码器层 = 注意力子层 + FFN 子层：
```python
x, cache = transformer_self_attention_layer(x, ...)  # 自注意力 + 交叉注意力
x = ffn(layer_norm(x)) + x                           # FFN + 残差
```

---

### Step 13 — `transformer_decoder()`
**文件**：`transformer.py` 第 2220 行（`def transformer_decoder` 函数）

多层解码器堆叠：循环调用 `transformer_decoder_layer()`，
最后输出 `layer_preprocess(x)`（最终 LayerNorm）。

---

### Step 14 — `transformer_decode()`
**文件**：`transformer.py` 第 174 行

解码器的顶层封装，与 `transformer_encode` 对应：
1. 施加 dropout
2. 调用 `decoder_function`
3. 返回 4D 张量（`expand_dims` 补维）

---

## 第五阶段：模型主体（Transformer 类）

> 目标：把编码器和解码器组合起来，理解完整的 forward pass。

### Step 15 — `Transformer.__init__()`
**文件**：`transformer.py` 第 267 行

了解类的结构：各函数引用（`_encoder_function`、`_decoder_function` 等）
允许子类通过替换函数指针来定制行为，无需重写整个前向传播。

---

### Step 16 — `Transformer.body()`  ⭐ 最重要
**文件**：`transformer.py` 第 348 行

这是模型的**核心主函数**，完整的 forward pass：

```
features["inputs"]
    ↓
self.encode()          → encoder_output, encoder_decoder_attention_bias
                                        ↓
features["targets"] → shift_right → self.decode() → decoder_output
```

训练时（teacher forcing）：解码器输入是右移后的真实目标序列。
推理时（自回归）：每步以上一步的预测结果为输入（见 Step 17-18）。

---

## 第六阶段：推理解码（Inference）

> 目标：理解训练结束后如何用模型生成序列。

### Step 17 — `_init_transformer_cache()`
**文件**：`transformer.py` 第 1172 行

为快速解码初始化 K/V 缓存结构：
```python
cache = {
    "layer_0": {"k": zeros, "v": zeros, "k_encdec": ..., "v_encdec": ...},
    "layer_1": {...},
    ...
    "encoder_output": ...,
}
```

每次解码只更新新增的 K/V，不重新计算历史部分。

---

### Step 18 — `fast_decode()` / `fast_decode_tpu()`
**文件**：`transformer.py` 第 1510 行（CPU/GPU 版）、约 1300 行（TPU 版）

自回归解码的主循环：
```python
while not finished:
    logits, cache = symbols_to_logits_fn(next_id, step, cache)
    next_id = sample(logits)   # 贪心 or 采样 or beam search
    decoded_ids.append(next_id)
```

CPU/GPU 版使用 `tf.while_loop` 动态形状；TPU 版预分配固定大小数组。

---

### Step 19 — `Transformer._beam_decode()` · `beam_search`
**文件**：`transformer.py` 第 501 行；`utils/beam_search.py`

束搜索：每步维护 `beam_size` 个候选序列，选择总得分最高的路径。
比贪心解码质量更高，是实际翻译系统常用的解码策略。

---

## 第七阶段：变体与扩展（选读）

> 目标：了解基础 Transformer 之上的各种扩展能力。

### TransformerEncoder（仅编码器）
**文件**：`transformer.py` 第 1750 行  
用于分类任务，去掉解码器，直接在编码器输出上接分类头。

### TransformerScorer（序列打分）
**文件**：`transformer.py` 第 1670 行  
不生成序列，而是对给定序列计算对数概率（困惑度评估、重排序）。

### TransformerMemory（循环记忆）
**文件**：`transformer.py` 第 2302 行 + `layers/transformer_memory.py`  
Transformer-XL 风格：通过缓存历史块的激活值处理超长序列。

### `update_hparams_for_tpu()`
**文件**：`transformer.py` 第 3245 行  
TPU 适配：切换为 Adafactor 优化器、固定 batch size、减少内存使用。

---

## 学习路径总览

```
Step 1-2   超参数         transformer_base_v1() → 知道模型有哪些参数
Step 3-5   注意力机制      attention_bias → multihead_attention → 位置编码
Step 6-9   编码器          prepare_encoder → ffn_layer → encoder → encode
Step 10-14 解码器          prepare_decoder → self_attention_layer → decoder → decode
Step 15-16 模型主体        Transformer.__init__ → body()          ← 核心 forward pass
Step 17-19 推理解码        init_cache → fast_decode → beam_decode
Step 20+   变体扩展        TransformerEncoder / Scorer / Memory
```

---

## 调试建议

1. **从 `transformer_test` 超参数开始**（2 层、hidden=16），模型小、运行快，适合打断点验证 shape。
2. **关注 tensor shape**：代码中频繁出现 `[batch, length, hidden]`（3D）和 `[batch, length, 1, hidden]`（4D），注意 t2t 框架要求 4D 格式。
3. **用 `attention_weights` 字典**可视化注意力，理解每层在关注什么。
4. **对比 v1/v2/v3**：三个版本的主要差异是 pre-norm vs post-norm 和学习率调度方式。
