# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer模型常用的可复用层。

本文件包含Transformer模型中频繁使用的三个核心函数：

1. transformer_prepare_encoder：编码器输入预处理
   - 添加位置编码（Positional Encoding）
   - 添加目标空间嵌入（Target Space Embedding）
   - 计算自注意力偏置（掩盖padding位置）
   - 支持packed数据集（多样本拼接）

2. transformer_encoder：多层编码器堆叠
   - 每层包含：自注意力 + 前馈网络（带残差连接和层归一化）
   - 支持pad_remover优化（跳过padding位置的计算）

3. transformer_ffn_layer：前馈网络层（FFN）
   - 支持多种FFN类型：dense_relu_dense（标准）、卷积型、MoE（混合专家）等
   - 这是Transformer "Add & Norm" 后面的关键组件

核心概念：
- padding：序列中用于对齐不同长度序列的填充位置（通常是0）
- attention bias：注意力偏置，用大负数(-1e9)屏蔽不应该attend to的位置
- packed dataset：将多个短序列拼接成一个长序列，提高GPU利用率
- pad_remover：训练时移除padding位置的技术，避免对padding做无效计算
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入注意力相关函数（多头注意力、位置编码、attention bias等）
from tensor2tensor.layers import common_attention
# 导入通用层函数（dropout、归一化、dense层等）
from tensor2tensor.layers import common_layers
# 导入专家工具（MoE混合专家模型相关）
from tensor2tensor.utils import expert_utils
# 导入MLPerf性能日志（标准基准测试日志，不影响模型功能）
from tensor2tensor.utils import mlperf_log

# 使用TF 1.x兼容接口
import tensorflow.compat.v1 as tf
# 导入estimator，用于获取ModeKeys（TRAIN/EVAL/PREDICT）
from tensorflow.compat.v1 import estimator as tf_estimator


# TODO(lukaszkaiser): 当不再需要时移除此函数。
def layers():
  """获取适用于TF1和TF2的layers模块（兼容性辅助函数）。

  返回 common_layers.layers()，该函数根据当前TF版本返回：
  - TF 1.x：tf.layers
  - TF 2.x：tf.keras.layers
  """
  return common_layers.layers()


def transformer_prepare_encoder(inputs, target_space, hparams, features=None,
                                type_ids=None, num_types=None,
                                reuse_target_embedding=tf.AUTO_REUSE):
  """为编码器准备输入（单分片，即单个GPU/TPU核心的数据）。

  该函数的核心工作是为编码器的自注意力机制准备以下内容：
  1. 处理padding mask（让注意力忽略padding位置）
  2. 添加目标空间嵌入（告诉编码器当前任务的目标类型，如翻译到英语还是法语）
  3. 添加位置编码（让模型知道每个token在序列中的位置）
  4. 可选地添加类型嵌入（如BERT的segment embedding，区分句子A和句子B）

  整体流程：
  ┌─────────────────┐
  │  inputs（词嵌入）│
  │       ↓        │
  │  + 目标空间嵌入  │  ← target_space_embedding（任务类型标识）
  │       ↓        │
  │  + 位置编码      │  ← timing signal（正弦/余弦 或 可学习嵌入）
  │       ↓        │
  │  + 类型嵌入      │  ← 可选，类似BERT segment embedding
  │       ↓        │
  │  encoder_input  │  → 送入编码器自注意力层
  └─────────────────┘

  Args:
    inputs: 输入张量，形状 [batch_size, input_length, hidden_dim]
            （由底层embedding层转换后的词向量矩阵）
    target_space: 标量整数张量，表示目标空间ID。
                  例如在多语言翻译中，ID=1表示翻译到英语，ID=2表示翻译到法语。
                  这个嵌入被加到所有输入位置，让编码器知道"翻译目标是什么"。
    hparams: 超参数对象，影响以下行为：
             - hparams.pos: 位置编码类型（"timing"/"emb"/"timing_from_features"）
             - hparams.proximity_bias: 是否添加近邻偏置（鼓励模型关注临近位置）
             - hparams.use_target_space_embedding: 是否使用目标空间嵌入
             - hparams.unidirectional_encoder: 是否使用单向编码器（因果mask）
    features: 可选的完整特征字典。"packed"数据集需要用到，
              其中包含 "inputs_segmentation"、"inputs_position"、
              "targets_segmentation" 等字段。
              - segmentation：每个token属于第几个原始样本（0=padding）
              - position：每个token在其原始样本内的位置（而不是拼接后的绝对位置）
    type_ids: 可选，形状 [batch, length] 的int64张量，用于类型嵌入。
              类似BERT的segment_ids，区分不同类型的token（如问题和答案）。
    num_types: 可选，类型嵌入的总类型数（必须与type_ids配合使用）。
    reuse_target_embedding: 是否复用已有的目标空间嵌入变量。
                            当输入和目标使用相同的模态（symbol modality）时有用。

  Returns:
    三元组 (encoder_input, encoder_self_attention_bias, encoder_decoder_attention_bias)：

    encoder_input: 处理后的编码器输入，形状 [batch_size, input_length, hidden_dim]
                   已添加了目标空间嵌入和位置编码。

    encoder_self_attention_bias: 编码器自注意力偏置，形状通常是
                                 [batch_size, 1, 1, input_length]
                                 非padding位置为0，padding位置为-1e9（大负数屏蔽）。
                                 对于packed数据集，还会屏蔽跨样本的注意力。

    encoder_decoder_attention_bias: 编码器-解码器注意力偏置，形状通常是
                                    [batch_size, 1, 1, input_length]
                                    用于解码器的交叉注意力，屏蔽输入的padding位置。
  """
  # 保存输入张量的静态形状（用于后续的嵌入维度计算）
  # shape.as_list() 返回如 [batch_size, length, hidden_dim] 的列表
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs  # 初始编码器输入就是词嵌入

  if features and "inputs_segmentation" in features:
    # ===== Packed数据集处理 =====
    # Packed数据集将多个原始序列拼接成一个长序列：
    # 原始：[A A A PAD] [B B PAD PAD]
    # Packed：[A A A B B] （segmentation=[1,1,1,2,2]，position=[0,1,2,0,1]）
    # 这样可以减少padding浪费，提高GPU/TPU利用率。
    inputs_segmentation = features["inputs_segmentation"]    # 每个token属于第几个原始样本
    inputs_position = features["inputs_position"]            # token在原始样本中的位置
    targets_segmentation = features["targets_segmentation"]  # 目标侧的segment ID

    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      # 单向编码器：每个位置只能attend to之前的位置（因果mask，下三角矩阵）
      # 用于GPT类语言模型或某些特殊任务
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      # 双向编码器（标准Transformer编码器）：
      # 使用 same_segment mask，同一原始样本内的token可以互相attend to
      # 不同原始样本之间（segmentation ID不同）不能互相attend to
      encoder_self_attention_bias = (
          common_attention.attention_bias_same_segment(
              inputs_segmentation, inputs_segmentation))

    # 编码器-解码器注意力偏置：
    # targets_segmentation 和 inputs_segmentation 对齐，
    # 第i个目标只能attend to第i个输入（同一原始样本对）
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(targets_segmentation,
                                                     inputs_segmentation))
  else:
    # ===== 普通数据集处理（每个样本独立，有padding） =====
    # embedding_to_padding: 判断哪些位置是padding（全零嵌入向量 → padding=1）
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    # attention_bias_ignore_padding: 将padding位置的偏置设为-1e9（屏蔽padding）
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)

    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      # 单向编码器：使用因果mask（下三角矩阵）
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      # 普通双向编码器：只需要屏蔽padding位置
      encoder_self_attention_bias = ignore_padding

    # 编码器-解码器注意力偏置（同样屏蔽输入中的padding）
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None  # 普通数据集不需要位置索引（按顺序添加位置编码）

  if hparams.proximity_bias:
    # 近邻偏置（Proximity Bias）：
    # 为自注意力添加额外的偏置，鼓励模型关注距离更近的位置
    # 偏置值 = sin(相对距离) 的某种形式，距离越近偏置越大
    # 适用于某些需要局部性先验的任务
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])

  if target_space is not None and hparams.get("use_target_space_embedding",
                                              True):
    # ===== 目标空间嵌入（Target Space Embedding） =====
    # 将一个标量"目标空间ID"转换为嵌入向量，添加到所有输入位置
    # 作用：告诉模型当前任务的"目标类型"
    # 例如在多语言翻译中：target_space=1→英语，2→法语，3→德语
    # 这样同一个编码器可以根据目标语言调整编码策略

    # common_layers.embedding: 查找嵌入表
    # - target_space: 嵌入索引（目标空间ID）
    # - 32: 嵌入表大小（32种可能的目标空间）
    # - ishape_static[-1]: 嵌入维度（与输入hidden_dim相同）
    emb_target_space = common_layers.embedding(
        target_space,
        32,                          # 32种目标空间（如多种语言）
        ishape_static[-1],           # 嵌入维度 = hidden_size
        name="target_space_embedding",
        dtype=hparams.get("activation_dtype", "float32"),
        reuse=reuse_target_embedding)

    # reshape: 标量嵌入变为 [1, 1, hidden_dim]，方便广播到 [batch, length, hidden_dim]
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    # 将目标空间嵌入加到所有输入位置（广播）
    encoder_input += emb_target_space

  # ===== 位置编码（Positional Encoding） =====
  # Transformer自注意力没有位置感知能力（对位置置换不变），
  # 因此必须显式添加位置信息
  if hparams.pos == "timing":
    # timing信号：正弦/余弦位置编码（原论文方案）
    # 公式：PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    #       PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    if inputs_position is not None:
      # packed数据集：使用给定的位置信息（而非0,1,2,...的顺序位置）
      encoder_input = common_attention.add_timing_signal_1d_given_position(
          encoder_input, inputs_position)
    else:
      # 普通数据集：顺序添加位置编码（位置0,1,2,...）
      encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  elif hparams.pos == "timing_from_features":
    # 从特征字典中获取位置信号（用于语音等特殊任务）
    # 例如音频帧的时间位置与文本token的序列位置不同
    encoder_input = common_attention.add_timing_signals_from_features(
        encoder_input, features, hparams.position_features)
  elif hparams.pos == "emb":
    # 可学习的位置嵌入（与词嵌入类似，位置也有可学习的向量）
    # 优点：更灵活；缺点：无法泛化到训练时未见过的位置
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "inputs_positional_embedding",
        inputs_position)

  # ===== 类型嵌入（Type Embeddings，可选） =====
  # 类似BERT的segment embedding，区分不同类型的token
  # 例如：问答任务中区分问题部分和答案部分（type_id=0为问题，1为答案）
  if type_ids is not None:
    if not num_types:
      raise ValueError("需要同时设置 num_types 参数。")
    # add_positional_embedding此处被复用为类型嵌入（按类型ID查表）
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, num_types, "inputs_type_embedding", type_ids)

  # 确保偏置张量的数据类型与输入一致（如bfloat16训练时需要转换）
  # cast_like: 将第一个参数的dtype转换为第二个参数的dtype
  encoder_self_attention_bias = common_layers.cast_like(
      encoder_self_attention_bias, encoder_input)
  encoder_decoder_attention_bias = common_layers.cast_like(
      encoder_decoder_attention_bias, encoder_input)

  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        attn_bias_for_padding=None):
  """多层Transformer编码器堆叠。

  将 N 个相同结构的编码器层（Encoder Layer）堆叠在一起，
  每层包含：
  1. 多头自注意力（Multi-Head Self-Attention）+ 残差连接 + LayerNorm
  2. 位置感知前馈网络（Position-wise FFN）+ 残差连接 + LayerNorm

  每个子层遵循 pre-norm 残差模式（v2/v3）或 post-norm 残差模式（v1）：
  - pre-norm：x = x + dropout(sublayer(LayerNorm(x)))
  - post-norm：x = LayerNorm(x + dropout(sublayer(x)))

  pad_remover 优化（非XLA编译时）：
  - 训练时，padding位置的计算完全没有意义（padding不携带信息）
  - PadRemover可以在FFN计算前移除padding位置，计算后恢复
  - 当序列较短时，这可以大幅减少FFN的计算量

  Args:
    encoder_input: 编码器输入张量，形状 [batch_size, input_length, hidden_dim]
                   （已添加位置编码）
    encoder_self_attention_bias: 自注意力偏置，形状 [batch_size, 1, 1, input_length]
                                 padding位置为-1e9，非padding为0
    hparams: 超参数对象，控制：
             - num_encoder_layers/num_hidden_layers：层数
             - num_heads：注意力头数
             - hidden_size：隐藏维度
             - attention_dropout：注意力权重dropout
             - filter_size：FFN中间层维度
             - use_pad_remover：是否使用pad_remover优化
    name: variable scope名称（默认"encoder"）
    nonpadding: 可选，形状 [batch_size, encoder_length] 的浮点张量
                1.0表示非padding位置，0.0表示padding位置。
                packed数据集必须传入此参数，否则从attention_bias推断。
                卷积层和pad_remover都需要这个mask。
    save_weights_to: 可选字典，用于存储注意力权重（供注意力可视化使用）。
                     字典的key是从variable scope自动生成的字符串。
    make_image_summary: 是否生成注意力权重的图像摘要（供TensorBoard可视化）。
    losses: 可选列表，训练时产生的额外损失（如MoE路由损失）会被追加到此列表。
    attn_bias_for_padding: 单向编码器时使用的padding偏置。
                           如果使用单向编码器（future mask），传入此参数。

  Returns:
    y: 编码器最终输出张量，形状 [batch_size, input_length, hidden_dim]
       已经过最后的LayerNorm处理。
  """
  x = encoder_input  # 当前层的输入（逐层更新）

  # 解析attention dropout的广播维度字符串（如"0,1"）
  # 广播dropout：同一batch/head内所有位置使用相同的dropout mask
  # 这比独立dropout更强的正则化效果
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  # 记录MLPerf性能日志（标准化基准测试用，不影响功能）
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    # ===== 准备pad_remover（padding移除器）=====
    if nonpadding is not None:
      # nonpadding直接传入时，padding = 1 - nonpadding
      padding = 1.0 - nonpadding
    else:
      # 从attention bias中推断padding位置
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        # 单向编码器使用特殊的padding偏置
        attention_bias = attn_bias_for_padding
      # attention_bias_to_padding: 将attention偏置转换为0/1的padding mask
      # 偏置值 < -1e8 的位置被认为是padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding

    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      # PadRemover：移除padding位置的优化器
      # 仅在非XLA编译时使用（TPU通常使用XLA编译，需要静态形状，不能用PadRemover）
      # 原理：FFN处理前将padding行删除，处理后在原位置填回零，避免无效计算
      pad_remover = expert_utils.PadRemover(padding)

    # ===== 逐层执行编码器层 =====
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        # ----- 子层1：多头自注意力 -----
        with tf.variable_scope("self_attention"):
          # 面积注意力（Area Attention）参数设置
          # 面积注意力：将相邻token聚合成"面积"后再做注意力（适合语音等任务）
          if layer < hparams.get("num_area_layers", 0):
            # 前几层使用面积注意力（宽/高可以>1）
            max_area_width = hparams.get("max_area_width", 1)
            max_area_height = hparams.get("max_area_height", 1)
            memory_height = hparams.get("memory_height", 1)
          else:
            # 其余层使用标准点积注意力（面积1x1）
            max_area_width = 1
            max_area_height = 1
            memory_height = 1

          # layer_preprocess：在子层前进行预处理（通常是LayerNorm）
          # multihead_attention：多头自注意力计算
          #   - memory=None 表示自注意力（Q=K=V，而非cross-attention）
          #   - encoder_self_attention_bias：屏蔽padding位置的偏置
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),  # LayerNorm预处理
              None,                          # memory=None→自注意力
              encoder_self_attention_bias,   # padding mask偏置
              hparams.attention_key_channels or hparams.hidden_size,  # key维度（0=用hidden_size）
              hparams.attention_value_channels or hparams.hidden_size,  # value维度
              hparams.hidden_size,           # 输出维度
              hparams.num_heads,             # 注意力头数
              hparams.attention_dropout,     # 注意力权重的dropout
              attention_type=hparams.self_attention_type,  # 注意力类型（点积/相对位置等）
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,       # 存储注意力权重（可视化用）
              make_image_summary=make_image_summary,  # 生成TensorBoard图像摘要
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              hard_attention_k=hparams.get("hard_attention_k", 0),   # 硬注意力top-k
              gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
              max_area_width=max_area_width,
              max_area_height=max_area_height,
              memory_height=memory_height,
              area_key_mode=hparams.get("area_key_mode", "none"),
              area_value_mode=hparams.get("area_value_mode", "none"),
              # 告知注意力层当前是否在训练（影响dropout等行为）
              training=(hparams.get("mode", tf_estimator.ModeKeys.TRAIN)
                        == tf_estimator.ModeKeys.TRAIN))
          # layer_postprocess：后处理（dropout + 残差连接，可选后归一化）
          x = common_layers.layer_postprocess(x, y, hparams)

        # ----- 子层2：位置感知前馈网络（FFN）-----
        with tf.variable_scope("ffn"):
          # layer_preprocess：LayerNorm预处理
          # transformer_ffn_layer：执行FFN（两层全连接，中间有ReLU激活）
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),  # LayerNorm预处理
              hparams,
              pad_remover,           # padding移除器（加速训练）
              conv_padding="SAME",   # 卷积类FFN的padding模式（编码器用SAME）
              nonpadding_mask=nonpadding,  # 非padding mask（卷积层需要）
              losses=losses)         # 额外损失列表（MoE路由损失等）
          # 残差连接
          x = common_layers.layer_postprocess(x, y, hparams)

    # ===== 最后的LayerNorm =====
    # 如果使用pre-norm（在每个子层前做归一化），
    # 最后一层的输出还没有被归一化，需要额外做一次归一化
    # 这步确保输出的数值范围合理
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x,
                          hparams,
                          pad_remover=None,
                          conv_padding="LEFT",
                          nonpadding_mask=None,
                          losses=None,
                          cache=None,
                          decode_loop_step=None,
                          readout_filter_size=0,
                          layer_collection=None):
  """Transformer中的前馈网络层（Position-wise Feed-Forward Network）。

  这是Transformer中的关键组件之一。在注意力层之后，每个位置
  独立地通过一个两层全连接网络：
  FFN(x) = max(0, x·W1 + b1)·W2 + b2
  其中 W1 的形状是 [hidden_size, filter_size]，
       W2 的形状是 [filter_size, hidden_size]

  支持多种FFN实现，通过 hparams.ffn_layer 控制：

  1. "dense_relu_dense"（默认，最常用）：
     hidden → filter_size（ReLU） → hidden
     可选 pad_remover 加速（跳过padding位置的计算）

  2. "conv_relu_conv"：
     使用卷积层代替全连接层，第一层使用大卷积核
     适合对局部特征敏感的任务

  3. "parameter_attention"：
     用参数注意力（Parameter Attention）代替FFN
     参数矩阵作为记忆，query是输入，注意力在参数上计算

  4. "conv_hidden_relu_with_sepconv"：
     深度可分离卷积（Depthwise Separable Convolution）
     第一层：3×1 卷积；第二层：31×1 卷积（类似扩展感受野）

  5. "sru"：
     Simple Recurrent Unit，简单循环单元
     在某些任务上比纯注意力更好

  6. "local_moe_tpu"、"local_moe"：
     局部混合专家（Mixture of Experts, MoE）
     多个专家网络并行，每个token只路由到部分专家
     可以在不增加计算的情况下增加模型容量

  7. "none"：
     跳过FFN（消融实验用）

  Args:
    x: 输入张量，形状 [batch_size, length, hparams.hidden_size]
    hparams: 超参数对象，关键字段：
             - ffn_layer: FFN类型（见上面的列表）
             - filter_size: FFN中间层维度（通常是hidden_size的4倍）
             - hidden_size: 输出维度
             - relu_dropout: ReLU激活后的dropout率
    pad_remover: 可选的 PadRemover 对象，用于跳过padding位置的计算。
                 仅对 "dense_relu_dense" 类型有效（速度优化）。
                 PadRemover.remove()：在处理前移除padding行
                 PadRemover.restore()：处理后在原位置填回零
    conv_padding: 卷积填充方式，"LEFT"（解码器因果卷积）或"SAME"（编码器）。
                  - "LEFT"：只在左侧填充，确保每个位置只看到之前的位置（因果性）
                  - "SAME"：两侧填充，保持序列长度不变（双向）
    nonpadding_mask: 可选，形状 [batch_size, length]，1.0=非padding，0.0=padding。
                     卷积类FFN需要此mask来避免在padding位置产生非零输出。
    losses: 可选列表，MoE的路由损失（负载均衡损失）会被追加到此列表。
            如果使用MoE层但 losses=None，会抛出 ValueError。
    cache: 可选字典，存储历史计算结果，用于快速解码。
           某些卷积类FFN需要缓存来实现增量计算。
    decode_loop_step: 可选整数，TPU解码时的当前步骤编号。
    readout_filter_size: 如果 > 0，用来覆盖 hparams.filter_size 作为中间层维度。
                         某些多任务学习设置中不同任务使用不同的filter_size。
    layer_collection: 可选，KFAC优化器的层集合。默认None（不使用KFAC）。

  Returns:
    输出张量，形状 [batch_size, length, hparams.hidden_size]

  Raises:
    ValueError: 当使用MoE层但 losses=None 时（无法收集路由损失）。
                当 ffn_layer 不是已知类型时。
  """
  ffn_layer = hparams.ffn_layer  # 从超参数获取FFN类型

  # 解析ReLU dropout的广播维度（逗号分隔的整数字符串）
  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))

  # 兼容处理：旧名称 "conv_hidden_relu" 等同于 "dense_relu_dense"
  if ffn_layer == "conv_hidden_relu":
    ffn_layer = "dense_relu_dense"

  if ffn_layer == "dense_relu_dense":
    # ===== 标准FFN：两层全连接 + ReLU（最常见的配置）=====
    # 结构：Linear(hidden→filter) → ReLU → Dropout → Linear(filter→hidden)

    # 记录MLPerf日志
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)

    if pad_remover:
      # PadRemover优化：移除padding行，节省计算
      original_shape = common_layers.shape_list(x)
      # reshape: [batch, length, hidden] → [batch*length, hidden]（展平）
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      # remove：移除padding行，x变为 [non_padding_count, hidden]
      x = tf.expand_dims(pad_remover.remove(x), axis=0)  # [1, non_padding, hidden]

    # dense_relu_dense：执行两层全连接
    # x: [batch_size, length, hidden_size] 或（使用pad_remover时）[1, non_padding, hidden]
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,    # 中间层维度（通常是hidden_size的4倍）
        hparams.hidden_size,    # 输出维度（恢复到hidden_size）
        dropout=hparams.relu_dropout,  # ReLU激活后的dropout
        dropout_broadcast_dims=relu_dropout_broadcast_dims,
        layer_collection=layer_collection)

    if pad_remover:
      # 恢复：将 [1, non_padding, hidden] 扩展回 [batch, length, hidden]
      # restore：在padding位置填回零
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output

  elif ffn_layer == "conv_relu_conv":
    # ===== 卷积FFN：两层卷积 + ReLU =====
    # 第一层：conv_first_kernel大小的卷积（通常3或1）
    # 第二层：1×1卷积（点卷积）
    # 支持快速解码缓存（通过cache参数）
    return common_layers.conv_relu_conv(
        x,
        readout_filter_size or hparams.filter_size,  # 中间层维度
        hparams.hidden_size,
        first_kernel_size=hparams.conv_first_kernel,  # 第一层卷积核大小
        second_kernel_size=1,                          # 第二层：1×1卷积
        padding=conv_padding,      # 填充方式（LEFT=解码器，SAME=编码器）
        nonpadding_mask=nonpadding_mask,  # padding mask
        dropout=hparams.relu_dropout,
        cache=cache,               # 快速解码缓存
        decode_loop_step=decode_loop_step)

  elif ffn_layer == "parameter_attention":
    # ===== 参数注意力FFN =====
    # 用参数矩阵作为"记忆"，输入作为query对参数矩阵做注意力
    # 参数矩阵是可学习的，类似于大的查找表
    return common_attention.parameter_attention(
        x,
        hparams.parameter_attention_key_channels or hparams.hidden_size,  # key维度
        hparams.parameter_attention_value_channels or hparams.hidden_size,  # value维度
        hparams.hidden_size,       # 输出维度
        readout_filter_size or hparams.filter_size,  # 参数矩阵行数（记忆大小）
        hparams.num_heads,         # 注意力头数
        hparams.attention_dropout)

  elif ffn_layer == "conv_hidden_relu_with_sepconv":
    # ===== 带深度可分离卷积的隐藏层ReLU =====
    # 第一层：3×1卷积（捕获局部特征）
    # 第二层：31×1卷积（大感受野，类似扩张卷积）
    # 适合语音处理等需要大感受野的任务
    return common_layers.conv_hidden_relu(
        x,
        readout_filter_size or hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),        # 第一层：3帧窗口
        second_kernel_size=(31, 1), # 第二层：31帧感受野
        padding="LEFT",            # 左填充（因果卷积）
        dropout=hparams.relu_dropout)

  elif ffn_layer == "sru":
    # ===== SRU：Simple Recurrent Unit（简单循环单元）=====
    # 用轻量级RNN代替FFN，适合需要序列记忆的任务
    return common_layers.sru(x)

  elif ffn_layer == "local_moe_tpu":
    # ===== 局部MoE（TPU版）：混合专家网络 =====
    # 多个专家FFN并行，每个token根据门控函数只激活部分专家
    # "local"表示专家在局部设备内（不需要all-reduce通信）
    # 优点：可以用2倍计算量达到更大的模型容量
    overhead = hparams.moe_overhead_eval
    if hparams.mode == tf_estimator.ModeKeys.TRAIN:
      overhead = hparams.moe_overhead_train  # 训练时使用不同的开销系数
    ret, loss = expert_utils.local_moe_tpu(
        x,
        hparams.filter_size // 2,   # 每个专家的中间层维度（总容量=filter_size/2×专家数）
        hparams.hidden_size,         # 输出维度
        hparams.moe_num_experts,     # 专家数量
        overhead=overhead,           # 开销系数（控制每个专家处理的token数）
        loss_coef=hparams.moe_loss_coef)  # 路由损失系数（负载均衡）

  elif ffn_layer == "local_moe":
    # ===== 局部MoE（CPU/GPU版）=====
    # ffn_expert_fn：创建每个专家的FFN函数
    overhead = hparams.moe_overhead_eval
    if hparams.mode == tf_estimator.ModeKeys.TRAIN:
      overhead = hparams.moe_overhead_train
    ret, loss = expert_utils.local_moe(
        x,
        True,   # is_training
        expert_utils.ffn_expert_fn(  # 每个专家的FFN函数
            hparams.hidden_size, [hparams.filter_size], hparams.hidden_size),
        hparams.moe_num_experts,   # 专家总数
        k=hparams.moe_k,           # 每个token激活的专家数（通常k=1或2）
        hparams=hparams)
    losses.append(loss)  # 将路由损失追加到损失列表
    return ret

  else:
    # ffn_layer == "none"：跳过FFN（用于消融实验）
    assert ffn_layer == "none"
    return x
