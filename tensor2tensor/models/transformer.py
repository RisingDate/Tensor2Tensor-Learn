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

"""Transformer模型，来自论文 "Attention Is All You Need"。

Transformer模型由编码器(Encoder)和解码器(Decoder)组成。两者都是
自注意力层(Self-Attention)和前馈网络层(Feed-Forward)的堆叠。
该模型在许多NLP任务，尤其是机器翻译上取得了优秀的效果。

详细模型描述和早期版本结果请参见：
"Attention Is All You Need" (https://arxiv.org/abs/1706.03762)

核心概念说明：
- Encoder（编码器）：将输入序列转化为连续表示（向量序列）
- Decoder（解码器）：利用编码器输出，自回归地生成输出序列
- Self-Attention（自注意力）：序列中每个位置与其他所有位置交互，建立全局依赖
- Multi-Head Attention（多头注意力）：并行运行多个注意力头，捕获不同子空间的信息
- Feed-Forward Network（前馈网络）：每个位置独立地通过两层全连接网络
- Positional Encoding（位置编码）：注入位置信息，因为注意力本身没有位置感知能力
- Layer Normalization（层归一化）：在每个子层前后进行归一化，稳定训练
- hparams（超参数）：控制模型结构和训练行为的各种参数集合
- beam search（束搜索）：一种贪心解码策略，同时维护多个候选序列
- cache（缓存）：快速解码时用于保存已计算的注意力键值对，避免重复计算
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# six.moves.range 提供了Python 2和3兼容的range函数
from six.moves import range  # pylint: disable=redefined-builtin

# 导入数据生成器（LibriSpeech是一个语音识别数据集）
from tensor2tensor.data_generators import librispeech
# 导入各种注意力机制相关函数（多头注意力、位置编码等）
from tensor2tensor.layers import common_attention
# 导入超参数基础设置
from tensor2tensor.layers import common_hparams
# 导入通用层函数（dropout、归一化、reshape等）
from tensor2tensor.layers import common_layers
# 导入不同的模态处理器（文本、图像、音频等）
from tensor2tensor.layers import modalities
# 导入Transformer特有的层（编码器、前馈层等）
from tensor2tensor.layers import transformer_layers
# 导入循环记忆机制（用于长序列处理）
from tensor2tensor.layers import transformer_memory
# 导入束搜索算法（解码时使用）
from tensor2tensor.utils import beam_search
# 导入专家工具（用于混合专家模型）
from tensor2tensor.utils import expert_utils
# 导入MLPerf日志（性能基准测试日志）
from tensor2tensor.utils import mlperf_log
# 导入注册表（用于注册模型和超参数，方便按名字查找）
from tensor2tensor.utils import registry
# 导入T2T模型基类
from tensor2tensor.utils import t2t_model

# 使用TF 1.x兼容接口
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

# pylint: disable=g-direct-tensorflow-import
# 原地操作（inplace ops），用于在TPU上高效更新张量
from tensorflow.python.ops import inplace_ops
# nest工具，用于操作嵌套的张量结构（如字典、列表的张量）
from tensorflow.python.util import nest
# pylint: enable=g-direct-tensorflow-import

# 为常用的层创建别名，方便在本文件及其他地方使用
# transformer_prepare_encoder: 准备编码器输入（添加位置编码、计算attention bias等）
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
# transformer_encoder: 执行编码器的多层自注意力+前馈网络
transformer_encoder = transformer_layers.transformer_encoder
# transformer_ffn_layer: 单个前馈网络层（两层全连接，中间有ReLU激活）
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


def transformer_encode(encoder_function, inputs, target_space, hparams,
                       attention_weights=None, features=None, losses=None,
                       prepare_encoder_fn=None, **kwargs):
  """对Transformer输入进行编码，返回编码器的输出表示。

  整个编码过程：
  1. 将4D输入张量展平为3D
  2. 调用prepare_encoder_fn准备编码器输入（添加位置编码、计算padding mask等）
  3. 对编码器输入施加dropout（训练时随机丢弃一些神经元，防止过拟合）
  4. 调用encoder_function执行多层自注意力+前馈网络
  5. 返回编码器输出和编码器-解码器注意力偏置

  Args:
    encoder_function: 编码器函数，接受编码器输入并返回编码输出
    inputs: Transformer输入张量，形状为 [batch_size, input_length, 1, hidden_dim]
            batch_size: 一批样本的数量
            input_length: 输入序列的长度（token数量）
            1: 空间维度（NLP中通常为1）
            hidden_dim: 每个token的向量表示维度
    target_space: 标量整数，目标空间ID（表示任务类型，如翻译到哪种语言）
    hparams: 超参数对象，包含hidden_size、num_heads等模型配置
    attention_weights: 可选字典，用于存储注意力权重（供可视化使用）
    features: 可选，完整的特征字典（"packed"数据集需要用到）
              "packed"数据集将多个短样本拼接成一个长序列以提高利用率
    losses: 可选列表，训练时的额外损失会被追加到此列表
    prepare_encoder_fn: 可选，替代默认的transformer_prepare_encoder函数
    **kwargs: 传递给encoder_function的额外关键字参数

  Returns:
    元组 (encoder_output, encoder_decoder_attention_bias)：
      encoder_output: 编码器输出表示，形状 [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: 编码器-解码器注意力偏置（用于遮蔽padding位置）
                                      形状 [batch_size, 1, 1, input_length]
  """
  # 将输入从4D [batch, length, 1, hidden] 展平为3D [batch, length, hidden]
  # 编码器和注意力函数通常需要3D输入
  inputs = common_layers.flatten4d3d(inputs)

  # 如果没有提供自定义的编码器准备函数，使用默认的
  if not prepare_encoder_fn:
    prepare_encoder_fn = transformer_prepare_encoder
  # 调用编码器准备函数，得到：
  # encoder_input: 添加了位置编码的输入
  # self_attention_bias: 自注意力偏置（用于遮蔽padding位置，形状 [batch, 1, 1, length]）
  # encoder_decoder_attention_bias: 编码器-解码器注意力偏置
  encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
      prepare_encoder_fn(
          inputs, target_space, hparams, features=features))

  # 记录MLPerf性能日志（用于标准化基准测试，不影响模型功能）
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  # 对编码器输入施加dropout
  # tf.nn.dropout(x, keep_prob): 以 (1-keep_prob) 的概率随机将元素置零
  # 这里 keep_prob = 1.0 - hparams.layer_prepostprocess_dropout
  # 训练时dropout有效（keep_prob < 1），推理时通常设为0（不dropout）
  encoder_input = tf.nn.dropout(encoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  # 单向编码器的padding偏置处理
  # 通常编码器是双向的（每个位置可以看到所有其他位置）
  # 如果设置了unidirectional_encoder，则使用因果mask，只能看到之前的位置
  attn_bias_for_padding = None
  # 否则编码器只使用encoder_self_attention_bias（来自padding）
  if hparams.unidirectional_encoder:
    attn_bias_for_padding = encoder_decoder_attention_bias

  # 执行编码器：多层自注意力 + 前馈网络的堆叠
  encoder_output = encoder_function(
      encoder_input,                           # 编码器输入（已加位置编码）
      self_attention_bias,                     # 自注意力偏置（掩盖padding）
      hparams,                                 # 超参数
      nonpadding=features_to_nonpadding(features, "inputs"),  # 非padding位置的mask
      save_weights_to=attention_weights,       # 保存注意力权重（用于可视化）
      make_image_summary=not common_layers.is_xla_compiled(),  # 是否生成注意力图像摘要
      losses=losses,                           # 额外损失列表
      attn_bias_for_padding=attn_bias_for_padding,  # 单向编码器的padding偏置
      **kwargs)

  return encoder_output, encoder_decoder_attention_bias


def transformer_decode(decoder_function,
                       decoder_input,
                       encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_self_attention_bias,
                       hparams,
                       attention_weights=None,
                       cache=None,
                       decode_loop_step=None,
                       nonpadding=None,
                       losses=None,
                       **kwargs):
  """从编码器表示中解码Transformer输出。

  整个解码过程：
  1. 对解码器输入施加dropout
  2. 调用decoder_function执行多层的：
     - 解码器自注意力（有因果mask，每个位置只能看到之前的位置）
     - 编码器-解码器交叉注意力（attend to编码器的输出）
     - 前馈网络
  3. 对输出进行reshape

  Args:
    decoder_function: 解码器函数（通常是transformer_decoder）
    decoder_input: 解码器的输入张量，即目标序列（右移一位），
                   形状 [batch_size, decoder_length, hidden_dim]
    encoder_output: 编码器输出，形状 [batch_size, input_length, hidden_dim]
                    解码器通过交叉注意力（cross-attention）读取这个输出
    encoder_decoder_attention_bias: 编码器-解码器注意力偏置，
                                    形状 [batch_size, 1, 1, input_length]
                                    用于遮蔽输入中的padding位置
    decoder_self_attention_bias: 解码器自注意力偏置，
                                 形状 [1, 1, decoder_length, decoder_length]
                                 包含因果mask（下三角矩阵），防止看到未来信息
    hparams: 超参数对象
    attention_weights: 可选字典，用于存储注意力权重（供可视化使用）
    cache: 可选字典，包含之前注意力步骤的键值对缓存，用于快速解码
           格式: {"layer_0": {"k": ..., "v": ...}, "layer_1": {...}, ...}
    decode_loop_step: 可选整数，解码循环的步骤编号，仅用于TPU推理
    nonpadding: 可选张量，形状 [batch_size, decoder_length]，标记非padding位置
    losses: 可选列表，训练时的额外损失
    **kwargs: 传递给decoder_function的额外关键字参数（如recurrent_memory等）

  Returns:
    解码器最终输出表示，形状 [batch_size, decoder_length, 1, hidden_dim]
    （t2t框架期望4D张量，所以在第3维添加了一个维度）
  """
  # 记录MLPerf日志
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)
  # 对解码器输入施加dropout（训练时随机失活部分神经元）
  decoder_input = tf.nn.dropout(decoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  # 执行解码器：多层的自注意力 + 交叉注意力 + 前馈网络
  decoder_output = decoder_function(
      decoder_input,                       # 解码器输入（已右移的目标序列+位置编码）
      encoder_output,                      # 编码器输出（交叉注意力的来源）
      decoder_self_attention_bias,         # 解码器自注意力偏置（因果mask）
      encoder_decoder_attention_bias,      # 编码器-解码器注意力偏置
      hparams,                             # 超参数
      cache=cache,                         # 键值对缓存（快速解码时使用）
      decode_loop_step=decode_loop_step,   # TPU解码步骤编号
      nonpadding=nonpadding,               # 非padding位置标记
      save_weights_to=attention_weights,   # 保存注意力权重
      losses=losses,                       # 额外损失
      **kwargs)

  # 判断是否在TPU上以XLA编译方式运行
  if (common_layers.is_xla_compiled() and
      hparams.mode == tf_estimator.ModeKeys.TRAIN):
    # TPU不喜欢多余的维度（XLA静态shape要求严格），训练时直接返回
    # TODO(noam): 等TPU更宽容后移除这段
    return decoder_output
  else:
    # t2t框架期望4D张量 [batch, length, height, channels]
    # 在axis=2处插入维度，将3D [batch, length, hidden] 变为4D [batch, length, 1, hidden]
    return tf.expand_dims(decoder_output, axis=2)


@registry.register_model  # 将这个类注册到t2t注册表，允许通过名字"transformer"引用
class Transformer(t2t_model.T2TModel):
  """Transformer模型 —— "Attention Is All You Need"的实现。

  这是经典的编码器-解码器Transformer架构：
  - 编码器：N层自注意力 + 前馈网络，处理源序列
  - 解码器：N层自注意力 + 交叉注意力 + 前馈网络，自回归地生成目标序列

  继承自T2TModel基类，提供了训练、推理等通用功能。
  """

  def __init__(self, *args, **kwargs):
    """初始化Transformer模型。

    设置各种函数引用，允许子类通过覆盖这些属性来定制行为，
    而无需重写整个forward pass。
    """
    super(Transformer, self).__init__(*args, **kwargs)
    # 存储注意力权重的字典，键为层名，值为权重张量（用于可视化注意力图）
    self.attention_weights = {}
    # 循环记忆（recurrent memory），用于处理超长序列（如Transformer-XL）
    # 默认为None，子类TransformerMemory会覆盖此属性
    self.recurrent_memory_by_layer = None
    # 编码器函数（可在子类中替换为自定义编码器）
    self._encoder_function = transformer_encoder
    # 解码器函数（可在子类中替换为自定义解码器）
    self._decoder_function = transformer_decoder
    # 初始化解码器缓存的函数（快速解码时使用）
    self._init_cache_fn = _init_transformer_cache
    # 准备编码器输入的函数（添加位置编码、计算attention bias等）
    self._prepare_encoder_fn = transformer_prepare_encoder
    # 准备解码器输入的函数（右移目标序列、添加位置编码、计算causal mask等）
    self._prepare_decoder_fn = transformer_prepare_decoder

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """对输入序列进行编码，返回编码器表示。

    这是对transformer_encode函数的包装，传入类实例的配置。

    Args:
      inputs: 输入张量，形状 [batch_size, input_length, 1, hidden_dim]
      target_space: 目标空间ID（表示翻译目标语言等）
      hparams: 超参数
      features: 完整特征字典（可选）
      losses: 额外损失列表（可选）

    Returns:
      (encoder_output, encoder_decoder_attention_bias) 元组
    """
    return transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,  # 传入注意力权重存储字典
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    """解码Transformer输出，详见transformer_decode函数说明。

    这是对transformer_decode函数的包装，传入类实例的配置。

    Args:
      decoder_input: 解码器输入（右移的目标序列）
      encoder_output: 编码器输出
      encoder_decoder_attention_bias: 编码器-解码器注意力偏置
      decoder_self_attention_bias: 解码器自注意力偏置（含因果mask）
      hparams: 超参数
      cache: 快速解码缓存
      decode_loop_step: TPU解码步骤
      nonpadding: 非padding位置标记
      losses: 额外损失列表
      **kwargs: 其他参数（如recurrent_memory_by_layer等）

    Returns:
      解码器输出表示，形状 [batch_size, decoder_length, 1, hidden_dim]
    """
    return transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
        **kwargs)

  def body(self, features):
    """Transformer模型的主体计算函数（前向传播）。

    这是模型的核心，执行完整的编码-解码过程：
    1. 如果有输入，编码输入序列
    2. 准备解码器输入（右移目标序列，添加位置编码）
    3. 解码
    4. 处理可能的额外损失

    Args:
      features: 特征字典，应包含：
        "inputs": Transformer输入，[batch_size, input_length, 1, hidden_dim]
                  对于纯语言模型（无输入）则不需要
        "targets": 目标解码器输出，[batch_size, decoder_length, 1, hidden_dim]
                   训练时这是真实目标序列（teacher forcing）
        "target_space_id": 标量int，表示目标空间（任务类型）

    Returns:
      最终解码器表示，形状 [batch_size, decoder_length, hidden_dim]
      如果有额外损失，返回 (output, {"extra_loss": ...}) 元组
    """
    hparams = self._hparams  # 获取超参数

    losses = []  # 收集训练过程中的额外损失（如MoE路由损失等）

    if self.has_input:
      # 有输入序列（如翻译任务的源语言）
      # 1. 准备编码器的输入
      inputs = self._prepare_inputs_for_body(features)
      target_space = features["target_space_id"]
      # 2. 编码输入序列 → 得到编码器表示
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      # 无输入序列（如语言模型，只有目标序列）
      encoder_output, encoder_decoder_attention_bias = (None, None)

    # 获取目标序列
    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)  # 保存原始形状，后面需要reshape回来
    # 将targets从4D [batch, length, 1, hidden] 展平为3D [batch, length, hidden]
    targets = common_layers.flatten4d3d(targets)
    # 准备解码器输入：
    # - 将targets右移一位（第一个位置用GO token/零向量填充）
    # - 添加位置编码
    # - 计算decoder_self_attention_bias（因果mask，下三角形式）
    decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
        targets, hparams, features=features)

    # 检查是否需要传递循环记忆参数
    # 不是所有Transformer子类都支持循环记忆，所以只在启用时传递
    decode_kwargs = {}
    if self.recurrent_memory_by_layer is not None:
      # chunk_number表示当前处理的"块"编号（用于循环记忆的状态管理）
      # 注意：chunk_number特征的形状与"targets"相同，是为了复用分片代码
      # 但实际上同一个样本内所有token应该有相同的chunk_number
      chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
      # 取每个样本第一个token的chunk_number作为该样本的chunk_number
      chunk_number_each_example = chunk_number_each_token[:, 0]
      # 可以取消注释下面的代码来验证同一批次中的token共享相同的chunk_number
      # with tf.control_dependencies([
      #     tf.assert_equal(chunk_number_each_token,
      #                     chunk_number_each_example[:, None])
      # ]):
      #   chunk_number_each_example = tf.identity(chunk_number_each_example)
      decode_kwargs = dict(
          recurrent_memory_by_layer=self.recurrent_memory_by_layer,
          chunk_number=chunk_number_each_example,
          )

    # 3. 解码：利用编码器输出和解码器输入，生成解码器隐藏状态
    decoder_output = self.decode(
        decoder_input,                     # 解码器输入（右移的目标序列+位置编码）
        encoder_output,                    # 编码器输出（用于交叉注意力）
        encoder_decoder_attention_bias,    # 编码器注意力偏置
        decoder_self_attention_bias,       # 解码器因果mask
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),  # 目标序列的非padding标记
        losses=losses,
        **decode_kwargs                    # 循环记忆相关参数（如有）
        )

    # 处理监督注意力损失（用于需要与期望注意力分布对齐的任务）
    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      # 计算实际注意力权重与期望注意力权重之间的损失
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    # 将解码器输出reshape回原始targets的形状
    # 这确保输出的batch维度和空间维度与输入保持一致
    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      # 如果有额外损失（如MoE路由损失），将其求和后一起返回
      # tf.add_n: 将列表中所有张量逐元素相加
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret

  def _prepare_inputs_for_body(self, features):
    """为模型主体准备输入。

    默认实现直接返回features["inputs"]。
    子类可以重写此方法来进行额外的输入预处理。

    Args:
      features: 特征字典，应包含 "inputs" 键，
                对应 Transformer 输入 [batch_size, input_length, 1, hidden_dim]

    Returns:
      将传给模型的输入，形状 [batch_size, input_length, 1, hidden_dim]
    """
    return features["inputs"]

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """贪心解码的快速版本（推理时使用）。

    贪心解码：每步选择概率最高的token，逐个生成输出序列。
    "快速"版本利用缓存避免重复计算，时间复杂度为O(n)而非O(n²)。

    对比慢速版本(_slow_greedy_infer)：
    - 慢速版：每步都重新计算所有位置的注意力
    - 快速版：缓存已计算的key/value，每步只计算新位置

    Args:
      features: 特征字典，键为字符串，值为Tensor
      decode_length: 整数，额外解码的时间步数
      use_tpu: 布尔值，是否为TPU构建推理图

    Returns:
      解码结果字典：
        "outputs": 整数张量，解码后的token ID
                   形状 [batch_size, <= decode_length]（beam_size=1时）
                   或 [batch_size, top_beams, <= decode_length]
        "scores": 束搜索的对数概率，贪心解码时为None

    Raises:
      NotImplementedError: 如果有多个数据分片
    """
    # 对于实数值模态（如回归输出）暂时使用慢速解码路径
    # 对于非点积注意力类型也使用慢速路径（缓存机制不兼容其他注意力类型）
    if (self._target_modality_is_real or
        self._hparams.self_attention_type != "dot_product"):
      return super(Transformer, self)._greedy_infer(features, decode_length)
    # 在模型的variable scope下执行（确保变量名正确）
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length)
      return self._fast_decode(features, decode_length)

  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """束搜索解码（推理时使用）。

    束搜索（Beam Search）：每步维护beam_size个最优候选序列，
    比贪心解码质量更高，但计算代价也更大。

    Args:
      features: 特征字典
      decode_length: 额外解码的时间步数
      beam_size: 束宽（同时维护的候选序列数量）
      top_beams: 最终返回的top-k个最优序列
      alpha: 长度惩罚系数（alpha越大，倾向于生成更长的序列）
             防止模型偏向短序列（因为短序列通常有更高的对数概率）
      use_tpu: 是否为TPU构建推理图

    Returns:
      解码结果字典：
        "outputs": 解码后的token ID
                   [batch_size, <= decode_length]（top_beams=1时）
                   或 [batch_size, top_beams, <= decode_length]
        "scores": 各序列的对数概率分数
    """
    # 只有点积注意力和相对位置点积注意力才能使用缓存（快速束搜索）
    if (self._hparams.self_attention_type not in [
        "dot_product", "dot_product_relative"
    ]):
      # 其他注意力类型不保证缓存机制正确工作，使用慢速版本
      return self._beam_decode_slow(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length, beam_size,
                                     top_beams, alpha)
      return self._fast_decode(features, decode_length, beam_size, top_beams,
                               alpha)

  def _prepare_inputs_for_decode(self, features):
    """为解码准备输入（推理时调用）。

    与训练时不同，解码时需要先经过模态底层处理（embedding等），
    才能送入编码器。

    Args:
      features: 特征字典

    Returns:
      处理后的输入张量（已经过模态底层处理，即词向量化等）
    """
    dp = self._data_parallelism  # 数据并行对象，用于多GPU/TPU并行
    hparams = self._hparams
    inputs = features["inputs"]
    # 整理输入形状（由于历史原因，这里有一些reshape操作）
    inputs = tf.expand_dims(inputs, axis=1)  # 在axis=1插入维度
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)  # 确保是5D张量
    s = common_layers.shape_list(inputs)
    # 将前两个维度合并（将数据分片展平）
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # 确保变量名与训练时一致（通过_shard_features）
    inputs = self._shard_features({"inputs": inputs})["inputs"]
    # 获取输入模态（如文本词表embedding、音频特征提取等）
    input_modality = self._problem_hparams.modality["inputs"]
    input_vocab_size = self._problem_hparams.vocab_size["inputs"]
    # 如果词表大小不是vocab_divisor的整数倍，补齐（TPU效率需要）
    if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
    # 获取模态名称
    modality_name = hparams.name.get("inputs",
                                     modalities.get_name(input_modality))(
                                         hparams, input_vocab_size)
    # 在模态对应的variable scope下，将输入ID转换为向量（embedding查表等）
    with tf.variable_scope(modality_name):
      bottom = hparams.bottom.get("inputs",
                                  modalities.get_bottom(input_modality))
      inputs = dp(bottom, inputs, hparams, input_vocab_size)
    return inputs

  def _fast_decode_tpu(self,
                       features,
                       decode_length,
                       beam_size=1,
                       top_beams=1,
                       alpha=1.0):
    """TPU上的快速解码。

    在TPU上实现贪心解码和束搜索，与CPU/GPU版本的主要区别：
    - TPU需要静态形状，所以decoded_ids预先分配固定大小
    - 使用inplace_ops进行原地更新，效率更高
    - 位置编码偏置需要静态切片

    Args:
      features: 特征字典
      decode_length: 额外解码的时间步数
      beam_size: 束宽（beam_size=1时等同于贪心解码）
      top_beams: 返回最优的前top_beams个序列
      alpha: 长度惩罚系数

    Returns:
      解码结果字典 {"outputs": ..., "scores": ...}

    Raises:
      NotImplementedError: 多数据分片不支持快速解码；
                           packed datasets不支持解码
    """
    if self._num_datashards != 1:
      raise NotImplementedError("快速解码只支持单个数据分片。")
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "打包数据集不支持解码。"
          " 如果要从数据集解码，请使用非打包版本的数据集。")
    dp = self._data_parallelism
    hparams = self._hparams
    # 获取目标模态（如文本词表的softmax层）
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    # 对齐词表大小到vocab_divisor的倍数（TPU矩阵乘法效率优化）
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor

    if self.has_input:
      # 有输入的情况（如翻译任务）
      inputs_shape = common_layers.shape_list(features["inputs"])
      if (target_modality == modalities.ModalityType.CLASS_LABEL or
          self._problem_hparams.get("regression_targets")):
        # 分类任务只需解码1个token
        decode_length = 1
      else:
        # 输出长度 = 输入长度 + 额外解码长度
        decode_length = (
            inputs_shape[1] + features.get("decode_length", decode_length))
      batch_size = inputs_shape[0]
      # 准备编码器输入并执行编码
      inputs = self._prepare_inputs_for_decode(features)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      # dp()返回一个列表（每个分片的输出），取第一个
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None  # 无部分目标（从头开始解码）
    else:
      # 无输入的情况（如语言模型，纯生成）
      encoder_output = None
      encoder_decoder_attention_bias = None

      # 准备部分目标（partial targets）：强制生成以这些token开头的序列
      # 可以来自 features["inputs"]（作为提示）或 features["targets"]
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      # 将partial_targets转换为2D整数张量 [batch_size, partial_length]
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      # 总解码长度 = 已有的部分目标长度 + 额外需要解码的长度
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    # 准备位置编码
    if hparams.pos == "timing":
      # 正弦/余弦位置编码（原始论文使用的方式）
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "timing_from_features":
      # 从特征中提取位置编码（用于语音等需要自定义位置信息的任务）
      positional_encoding = common_attention.add_timing_signals_from_features(
          tf.zeros([1, decode_length + 1, hparams.hidden_size]), features,
          hparams.position_features)
    elif hparams.pos == "emb":
      # 可学习的位置嵌入（类似词向量，每个位置有独立的可训练向量）
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length + 1, hparams.hidden_size]),
          hparams.max_length, "body/targets_positional_embedding", None)
    else:
      # 不使用位置编码（如相对位置注意力，位置信息已内置在注意力中）
      positional_encoding = None

    def preprocess_targets(targets, i):
      """对目标token进行预处理，将token ID转换为模型需要的向量表示。

      在每步解码时，将上一步生成的token ID经过以下处理：
      1. 词向量查表（embedding lookup）：ID → 向量
      2. 展平为3D
      3. 第0步用零向量替代（GO token，解码开始符号）
      4. 添加位置编码

      Args:
        targets: token ID张量，形状 [batch_size, 1]
        i: 当前解码步骤编号（整数）

      Returns:
        处理后的目标向量，形状 [batch_size, 1, hidden_dim]
      """
      # 通过_shard_features确保变量名与训练时一致
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        # dp(bottom, ...): 使用数据并行执行bottom（词向量化等）
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      # 将4D展平为3D [batch, 1, hidden]
      targets = common_layers.flatten4d3d(targets)

      # 第0步时用零向量替代实际的token向量
      # 原因：transformer_prepare_decoder会将目标右移一位，
      # 第0个位置填充的是GO token（全零向量）
      # tf.cond(condition, true_fn, false_fn): 条件分支，TF图模式下的if-else
      # tf.equal(i, 0): 检查i是否等于0
      # tf.zeros_like(targets): 生成与targets形状相同的全零张量
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        positional_encoding_shape = positional_encoding.shape.as_list()
        # 提取第i步的位置编码，添加到目标向量上
        # tf.slice(input, begin, size): 从begin位置开始切取size大小的片段
        targets += tf.slice(
            positional_encoding, [0, i, 0],
            [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
      return targets

    # 创建解码器自注意力偏置（因果mask，下三角矩阵）
    # attention_bias_lower_triangle: 对角线以上的位置为-inf，防止看到未来信息
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      # 近邻偏置：距离越近的token注意力分数越高（使用相对位置的正弦函数）
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_tpu_fn(ids, i, cache):
      """将token ID转换为下一步的logits（用于TPU推理）。

      这是解码循环的核心函数，每步执行：
      1. 取最新的token ID
      2. 进行目标预处理（embedding + 位置编码）
      3. 运行解码器（用缓存避免重复计算）
      4. 通过顶层模态计算logits
      5. 处理强制部分目标

      Args:
        ids: token ID张量，形状 [batch_size, current_length] 或 [batch_size*beam_size, ...]
        i: 当前解码步骤（TPU上是静态整数）
        cache: 键值对缓存字典

      Returns:
        (ret, cache): ret是下一个token的logits，形状 [batch_size, vocab_size]
                      cache是更新后的缓存
      """
      # 只取最后一个token（当前步骤要预测的位置）
      ids = ids[:, -1:]
      # 在axis=2和axis=3扩展维度，变成 [batch, 1, 1, 1]（符合t2t格式）
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      # 预处理：embedding + 位置编码
      targets = preprocess_targets(targets, i)

      # TPU需要静态形状，用tf.slice而非动态切片
      bias_shape = decoder_self_attention_bias.shape.as_list()
      # 提取第i行（当前位置对所有位置的注意力偏置）
      bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                      [bias_shape[0], bias_shape[1], 1, bias_shape[3]])

      with tf.variable_scope("body"):
        # 运行解码器的一步（利用缓存避免重复计算历史key/value）
        body_outputs = dp(
            self.decode,
            targets,                                  # 当前步的输入向量
            cache.get("encoder_output"),              # 编码器输出（cross-attention用）
            cache.get("encoder_decoder_attention_bias"),  # 编码器attention偏置
            bias,                                     # 当前步的因果mask
            hparams,
            cache,                                    # 缓存（含历史key/value）
            i,                                        # 步骤编号（TPU用）
            nonpadding=features_to_nonpadding(features, "targets"))

      # 获取模态名称（用于variable scope）
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets",
                              modalities.get_top(target_modality))
        # 通过顶层模态（通常是线性变换+softmax）计算词表logits
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

      # 去掉多余的维度，变为 [batch_size, vocab_size]
      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # 如果当前步骤在partial_targets范围内，强制logits指向该token
        # 方法：用one-hot向量（指定位置极大，其他位置极小负数）替换logits
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          # tf.one_hot(indices, depth, on_value, off_value):
          # 在indices位置为on_value(0.0)，其他位置为off_value(-1e9)
          # 注意：这里用0.0作为"选中"值，-1e9作为"未选中"值
          # softmax后几乎所有概率集中在one-hot位置
          return tf.one_hot(
              tf.tile(
                  tf.slice(partial_targets, [0, i],
                           [partial_targets.shape.as_list()[0], 1]),
                  [beam_size]), vocab_size, 0.0, -1e9)

        # tf.less(i, partial_targets_length): 判断是否还在partial_targets范围内
        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    # 获取结束符ID（eos = end of sequence）
    eos_id = self.get_decode_end_id() or beam_search.EOS_ID
    # 采样温度：0表示贪心选取最高概率，>0时引入随机性
    temperature = features.get("sampling_temp",
                               getattr(hparams, "sampling_temp", 0.0))
    # top-k采样：只从概率最高的k个token中采样（-1表示不限制）
    top_k = features.get("sampling_keep_top_k",
                         getattr(hparams, "sampling_keep_top_k", -1))

    # 执行快速解码（TPU版本）
    ret = fast_decode_tpu(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_tpu_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        init_cache_fn=self._init_cache_fn,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length,
        eos_id=eos_id,
        sampling_temperature=temperature,
        top_k=top_k)
    if partial_targets is not None:
      # 去掉partial_targets部分，只保留新生成的内容
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret

  def get_decode_start_id(self):
    """返回解码器第一个输入符号的ID（即GO token的ID）。

    默认情况下返回None，会被映射为0（零向量）。
    子类可以覆盖此方法来使用不同的开始符号。
    返回的ID用于查询embedding矩阵，得到解码器第一个输入向量。

    Returns:
      GO token的ID，默认为None（映射到0）
    """
    return None

  def get_decode_end_id(self):
    """返回终止解码的输出符号ID（即EOS token的ID）。

    当解码到此ID时，认为序列生成完毕，停止解码。
    子类可以覆盖此方法来使用不同的结束符号。

    Returns:
      EOS token的ID，默认为None（使用beam_search.EOS_ID=1）
    """
    return None

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0,
                   preprocess_targets_method=None):
    """快速解码（CPU/GPU版本）。

    同时实现贪心解码和束搜索：
    - beam_size=1时等同于贪心解码
    - beam_size>1时执行束搜索

    与TPU版本的区别：
    - CPU/GPU版本使用动态形状，不需要预先分配固定大小
    - 使用tf.concat动态扩展decoded_ids

    Args:
      features: 特征字典
      decode_length: 额外解码的时间步数
      beam_size: 束宽
      top_beams: 返回最优的前top_beams个序列
      alpha: 长度惩罚系数
      preprocess_targets_method: 目标预处理方法，None时使用内部定义的preprocess_targets

    Returns:
      解码结果字典 {"outputs": ..., "scores": ..., "cache": ...}

    Raises:
      NotImplementedError: 多数据分片时抛出
    """
    if self._num_datashards != 1:
      raise NotImplementedError("快速解码只支持单个数据分片。")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "打包数据集不支持解码。"
          " 如果要从数据集解码，请使用非打包版本的数据集。")
    if self.has_input:
      inputs_shape = common_layers.shape_list(features["inputs"])
      if (target_modality == modalities.ModalityType.CLASS_LABEL or
          self._problem_hparams.get("regression_targets")):
        decode_length = 1
      else:
        decode_length = (
            inputs_shape[1] + features.get("decode_length", decode_length))
      batch_size = inputs_shape[0]
      inputs = self._prepare_inputs_for_decode(features)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      # 检查是否有partial_targets（训练时可以指定一个前缀，强制模型生成以此开头的序列）
      partial_targets = features.get("partial_targets")
    else:
      # 无输入的纯生成模型
      encoder_output = None
      encoder_decoder_attention_bias = None

      # 从inputs或targets中获取partial_targets（提示词）
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None

    if partial_targets is not None:
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    # 准备位置编码（各种方式：定时信号、特征驱动、可学习嵌入）
    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "timing_from_features":
      positional_encoding = common_attention.add_timing_signals_from_features(
          tf.zeros([1, decode_length, hparams.hidden_size]), features,
          hparams.position_features)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length, hparams.hidden_size]), hparams.max_length,
          "body/targets_positional_embedding", None)
    else:
      positional_encoding = None

    def preprocess_targets(targets, i):
      """对目标token进行预处理（CPU/GPU版本）。

      将token ID经过embedding、展平和位置编码处理。

      Args:
        targets: token ID张量，形状 [batch_size, 1]
        i: 标量，当前解码步骤编号

      Returns:
        处理后的目标向量，形状 [batch_size, 1, hidden_dim]
      """
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)

      # 第0步时用零向量（GO token表示开始）
      if not self.get_decode_start_id():
        targets = tf.cond(
            tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        # CPU/GPU版本使用动态切片，位置编码从第i位置取一个
        targets += positional_encoding[:, i:i + 1]
      return targets

    # 创建因果mask（下三角注意力偏置）
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    # 创建用于保存编码器-解码器注意力历史的缓存（用于可视化）
    att_cache = {"attention_history": {}}
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
    if encoder_output is not None:
      att_batch_size, enc_seq_length = common_layers.shape_list(
          encoder_output)[0:2]
      # 为每一层初始化空的注意力历史张量
      for layer in range(num_layers):
        att_cache["attention_history"]["layer_%d" % layer] = tf.zeros(
            [att_batch_size, hparams.num_heads, 0, enc_seq_length])

    def update_decoder_attention_history(cache):
      """将当前步骤的注意力权重追加到历史记录中（用于可视化）。

      遍历self.attention_weights中属于解码器交叉注意力的权重
      （包含"decoder"但不包含"self"和"logits"的键），
      将其追加到cache["attention_history"]对应层的张量中。
      """
      # 筛选出解码器的编码器-解码器注意力权重（cross-attention）
      for k in [x for x in self.attention_weights
                if "decoder" in x and "self" not in x and "logits" not in x]:
        # 从键名中找到"layer_"的位置
        idx = k.find("layer_")
        if idx < 0:
          continue
        # 提取层号（形如"layer_0"、"layer_1"等）
        layer_nbr = k[idx + 6:]
        idx = 0
        # 解析层号字符串为整数
        while idx + 1 < len(layer_nbr) and layer_nbr[:idx + 1].isdigit():
          idx += 1
        layer_nbr = "layer_%d" % int(layer_nbr[:idx])
        if layer_nbr in cache["attention_history"]:
          # 将当前步骤的注意力权重沿时间轴(axis=2)拼接到历史记录
          cache["attention_history"][layer_nbr] = tf.concat(
              [cache["attention_history"][layer_nbr],
               self.attention_weights[k]],
              axis=2)

    # 如果没有自定义的预处理方法，使用内部定义的
    if not preprocess_targets_method:
      preprocess_targets_method = preprocess_targets

    def symbols_to_logits_fn(ids, i, cache):
      """将token ID转换为下一步的logits（CPU/GPU版本）。

      每步解码的核心函数，执行以下操作：
      1. 取最新的token ID
      2. 进行目标预处理（embedding + 位置编码）
      3. 运行解码器（利用缓存避免重复计算）
      4. 通过顶层模态计算logits
      5. 处理强制部分目标（partial targets）

      Args:
        ids: token ID张量，形状 [batch_size, current_length]
        i: 标量，当前解码步骤
        cache: 键值对缓存字典

      Returns:
        (ret, cache): ret是下一个token的logits [batch_size, vocab_size]
                      cache是更新后的缓存
      """
      # 只取序列最后一个ID（当前要预测的位置）
      ids = ids[:, -1:]
      # 添加维度使其符合t2t格式：[batch, 1] → [batch, 1, 1, 1]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      # 预处理：embedding + 位置编码
      targets = preprocess_targets_method(targets, i)

      # 动态切片因果mask（CPU/GPU支持动态形状）
      # decoder_self_attention_bias形状：[1, 1, decode_length, decode_length]
      # 取第i行，即当前位置对前i+1个位置的注意力偏置
      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,                                        # 当前步骤的输入
            cache.get("encoder_output"),                    # 编码器输出
            cache.get("encoder_decoder_attention_bias"),    # 编码器注意力偏置
            bias,                                           # 当前步的因果mask
            hparams,
            cache,                                          # 历史key/value缓存
            nonpadding=features_to_nonpadding(features, "targets"))

      # 更新注意力可视化缓存
      update_decoder_attention_history(cache)

      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets", modalities.get_top(target_modality))
        # 通过顶层网络计算词表logits（线性变换映射到词表大小）
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

      # 去掉多余维度，变为 [batch_size, vocab_size]
      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # 如果当前步在partial_targets范围内，强制logits指向该token
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          # 构造强制logits：在partial_targets[i]的位置为0.0，其他位置为-1e9
          # 注意：这里partial_targets[:, i]取第i列（第i个时间步的所有batch的token）
          # tf.tile(..., [beam_size]): 沿最后一维重复beam_size次（束搜索需要）
          return tf.one_hot(
              tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
              -1e9)

        # 在partial_targets范围内使用强制logits，否则使用正常logits
        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    # 获取开始符（SOS）和结束符（EOS）的ID
    sos_id = self.get_decode_start_id() or 0   # Start Of Sequence token ID
    eos_id = self.get_decode_end_id() or beam_search.EOS_ID  # End Of Sequence token ID
    # 采样温度（0=贪心，>0引入随机性）
    temperature = features.get("sampling_temp",
                               getattr(hparams, "sampling_temp", 0.0))
    # top-k采样参数（-1表示不限制）
    top_k = features.get("sampling_keep_top_k",
                         getattr(hparams, "sampling_keep_top_k", -1))

    # 执行快速解码
    ret = fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        init_cache_fn=self._init_cache_fn,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length,
        sos_id=sos_id,
        eos_id=eos_id,
        sampling_temperature=temperature,
        top_k=top_k,
        cache=att_cache)  # 传入注意力历史缓存
    if partial_targets is not None:
      # 去掉partial_targets部分，只保留新生成的内容
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret


def _init_transformer_cache(cache, hparams, batch_size, attention_init_length,
                            encoder_output, encoder_decoder_attention_bias,
                            scope_prefix):
  """创建Transformer快速解码的初始缓存。

  快速解码（incremental decoding）的关键优化：
  - 在自注意力中，每个新token需要attend to所有之前的token
  - 如果每步都重新计算，复杂度为O(n²)
  - 通过缓存key和value，每步只需计算新token的部分，复杂度降为O(n)

  缓存结构：
  {
    "layer_0": {
      "k": 自注意力的key缓存，形状 [batch, num_heads, length, depth_per_head]
      "v": 自注意力的value缓存，形状 [batch, num_heads, length, depth_per_head]
      "k_encdec": 编码器-解码器注意力的key（来自编码器，不变）
      "v_encdec": 编码器-解码器注意力的value（来自编码器，不变）
    },
    "layer_1": {...},
    ...
    "encoder_output": 编码器输出（所有层共用）
    "encoder_decoder_attention_bias": 编码器注意力偏置
  }

  Args:
    cache: 已有的缓存字典（如果为None则创建新的）
    hparams: 超参数对象
    batch_size: 批次大小
    attention_init_length: 初始注意力序列长度（TPU版为decode_length，CPU/GPU版为0）
    encoder_output: 编码器输出张量，形状 [batch, input_length, hidden_dim]
                    如果为None则不初始化编码器相关的缓存
    encoder_decoder_attention_bias: 编码器-解码器注意力偏置
    scope_prefix: 解码器层变量作用域的前缀字符串（通常是"body/"）

  Returns:
    初始化好的缓存字典
  """
  # key通道数（用于注意力计算的key的维度）
  key_channels = hparams.attention_key_channels or hparams.hidden_size
  # value通道数
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  # 解码器层数（如果没有单独指定，使用num_hidden_layers）
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  # 是否使用3D注意力变量（将多头权重存储为[heads, depth, depth]的3D矩阵）
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  if cache is None:
    cache = {}
  # 为每一层初始化自注意力的key/value缓存
  cache.update({
      "layer_%d" % layer: {  # pylint: disable=g-complex-comprehension
          "k":  # key缓存：初始为全零，形状 [batch, heads, length, depth_per_head]
              common_attention.split_heads(
                  tf.zeros([batch_size,
                            attention_init_length,
                            key_channels]), hparams.num_heads),
          "v":  # value缓存：初始为全零
              common_attention.split_heads(
                  tf.zeros([batch_size,
                            attention_init_length,
                            value_channels]), hparams.num_heads),
      } for layer in range(num_layers)
      # 这是一个字典推导式：为每一层（0到num_layers-1）创建一个子字典
  })

  # 对于某些FFN层类型，还需要缓存前馈网络的中间结果
  # 如果ffn_layer是"dense_relu_dense"或"conv_hidden_relu"，则不需要缓存"f"
  # 否则需要初始化"f"缓存，避免beam search时因形状不一致出错
  if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
    for layer in range(num_layers):
      cache["layer_%d" % layer]["f"] = tf.zeros(
          [batch_size, 0, hparams.hidden_size])

  if encoder_output is not None:
    # 预计算编码器-解码器注意力的key和value
    # 由于编码器输出在解码过程中不变，这些key/value可以一次性计算并缓存
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      # 在对应的variable scope下计算（确保变量名与训练时一致）
      with tf.variable_scope(
          "%sdecoder/%s/encdec_attention/multihead_attention" %
          (scope_prefix, layer_name)):
        # 计算编码器输出作为注意力key的变换
        k_encdec = common_attention.compute_attention_component(
            encoder_output,
            key_channels,
            name="k",
            vars_3d_num_heads=vars_3d_num_heads)
        # split_heads: 将 [batch, length, num_heads * depth] 分成多头形式
        #              变为 [batch, num_heads, length, depth_per_head]
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
        # 计算编码器输出作为注意力value的变换
        v_encdec = common_attention.compute_attention_component(
            encoder_output,
            value_channels,
            name="v",
            vars_3d_num_heads=vars_3d_num_heads)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
      # 将预计算的编码器key/value存入每层的缓存
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    # 将编码器输出和偏置也存入缓存（供解码步骤使用）
    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
  return cache


def fast_decode_tpu(encoder_output,
                    encoder_decoder_attention_bias,
                    symbols_to_logits_fn,
                    hparams,
                    decode_length,
                    vocab_size,
                    init_cache_fn=_init_transformer_cache,
                    beam_size=1,
                    top_beams=1,
                    alpha=1.0,
                    sos_id=0,
                    eos_id=beam_search.EOS_ID,
                    batch_size=None,
                    force_decode_length=False,
                    scope_prefix="body/",
                    use_top_k_with_unique=True,
                    sampling_temperature=0.0,
                    top_k=-1):
  """给定编码器输出，在TPU上执行快速解码。

  TPU版本与CPU/GPU版本的主要区别：
  - TPU要求静态形状：decoded_ids预先分配 [batch, decode_length] 的固定大小
  - 使用 inplace_ops.alias_inplace_update 进行原地更新（比tf.concat更高效）
  - 使用固定形状的 shape_invariants，不需要 [None, None] 这样的动态形状
  - 束搜索使用 use_top_k_with_unique 选项（TPU优化的top-k）

  实现了贪心解码和束搜索：
  - beam_size=1：贪心解码，每步选择最高概率的token
  - beam_size>1：束搜索，同时维护多个候选序列

  Args:
    encoder_output: 编码器输出张量
    encoder_decoder_attention_bias: 编码器-解码器注意力偏置
    symbols_to_logits_fn: 增量解码函数，映射 (ids, step, cache) → logits
    hparams: 超参数
    decode_length: 解码长度（TPU上必须是静态值）
    vocab_size: 输出词表大小
    init_cache_fn: 初始化缓存的函数
    beam_size: 束宽
    top_beams: 返回最优的前top_beams个序列
    alpha: 长度惩罚系数
    sos_id: 开始符号ID
    eos_id: 结束符号ID
    batch_size: 批次大小（无输入时必须显式传入）
    force_decode_length: 是否强制解码完整的decode_length步（即使遇到eos也不停止）
    scope_prefix: 解码器变量作用域前缀
    use_top_k_with_unique: 是否使用TPU优化的快速top-k（精度略低但速度快）
    sampling_temperature: 采样温度
    top_k: top-k采样的k值

  Returns:
    解码结果字典 {"outputs": decoded_ids, "scores": scores}
      outputs形状：[batch_size, decode_length]（top_beams=1）
                 或 [batch_size, top_beams, decode_length]

  Raises:
    NotImplementedError: 目前TPU版本不支持partial targets与beam_size>1的组合
  """
  if encoder_output is not None:
    # 从编码器输出推断批次大小
    batch_size = common_layers.shape_list(encoder_output)[0]

  # 初始化解码缓存（TPU版本：attention_init_length=decode_length，预分配完整大小）
  cache = init_cache_fn(None, hparams, batch_size, decode_length,
                        encoder_output, encoder_decoder_attention_bias,
                        scope_prefix)

  # 记录MLPerf束搜索日志
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
      value={
          "vocab_size": vocab_size,
          "batch_size": batch_size,
          "beam_size": beam_size,
          "alpha": alpha,
          "max_decode_length": decode_length
      },
      hparams=hparams)
  if beam_size > 1:  # 执行束搜索
    # 初始ID：所有批次的起始token都是sos_id
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    # 执行束搜索
    # decoded_ids形状：[batch, beam_size, decode_length+1]（包含初始sos token）
    # scores形状：[batch, beam_size]
    # _：更新后的缓存（这里忽略）
    decoded_ids, scores, _ = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1),      # 如果只需要最优序列，早停优化
        use_tpu=True,                     # 使用TPU优化版本
        use_top_k_with_unique=use_top_k_with_unique)

    if top_beams == 1:
      # 只需要最优序列：取第0个beam，跳过初始sos token（索引1:）
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      # 返回前top_beams个序列
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # 贪心解码（beam_size=1）

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """贪心解码的单步循环。

      每步执行：
      1. 调用symbols_to_logits_fn得到下一步的logits
      2. 计算log概率
      3. 根据采样策略选择下一个token
      4. 更新已生成序列和累计对数概率
      5. 检查是否到达EOS

      Args:
        i: 当前步骤编号（循环变量）
        hit_eos: 布尔张量 [batch_size]，标记哪些序列已到达EOS
        next_id: 上一步生成的token ID，形状 [batch_size, 1]
        decoded_ids: 已生成的序列，形状 [batch_size, decode_length]（TPU预分配固定大小）
        cache: 注意力键值对缓存
        log_prob: 累计对数概率，形状 [batch_size]

      Returns:
        (i+1, hit_eos, next_id, decoded_ids, cache, log_prob) 更新后的循环变量
      """
      # 获取当前步骤的logits和更新后的缓存
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      # 将logits转换为log概率（更数值稳定）
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = sampling_temperature
      if hparams.sampling_method == "random_per_example":
        # 每个样本使用独立的采样温度（适用于批次内需要不同多样性的场景）
        next_id = common_layers.sample_temperature_per_example(
            logits, temperature, top_k)
      else:
        if hparams.sampling_method == "argmax":
          temperature = 0.0  # argmax即温度为0的贪心选取
        # 用温度采样（temperature=0时等同于argmax）
        next_id = common_layers.sample_with_temperature(logits, temperature,
                                                        top_k)

      # 构建索引：[[0, token_0], [1, token_1], ...] 用于gather_nd
      log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
                                  axis=1)
      # 累加当前步骤的log概率
      # (1 - tf.to_float(hit_eos)): 已到达EOS的序列不再累加概率
      # 注意：这里刻意在累加之后再更新hit_eos，
      # 这样包含了第一个EOS token的概率，但不包含EOS之后的token
      log_prob += tf.gather_nd(
          log_probs, log_prob_indices) * (1 - tf.to_float(hit_eos))
      # 更新hit_eos标记（用|=即逻辑OR，一旦到达EOS就保持True）
      hit_eos |= tf.equal(next_id, eos_id)

      # 更新decoded_ids（TPU版本：原地更新预分配的固定大小张量）
      next_id = tf.expand_dims(next_id, axis=1)  # [batch] → [batch, 1]
      # 转置后进行原地更新（inplace update效率高于tf.concat）
      decoded_ids = tf.transpose(decoded_ids)  # [batch, len] → [len, batch]
      decoded_ids = inplace_ops.alias_inplace_update(
          decoded_ids, i, tf.squeeze(next_id, axis=1))  # 更新第i个位置
      decoded_ids = tf.transpose(decoded_ids)  # 转置回来
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      """判断解码循环是否应该继续。

      当以下任一条件成立时停止：
      1. 已达到最大解码长度 decode_length
      2. 所有序列都已到达EOS（force_decode_length=False时）

      Args:
        i: 当前步骤
        hit_eos: 布尔张量，标记每个序列是否已到达EOS
        *_: 其他循环变量（不需要用到）

      Returns:
        布尔标量，True表示继续循环，False表示停止
      """
      finished = i >= decode_length
      if not force_decode_length:
        # tf.reduce_all: 所有元素都为True才返回True（即所有序列都到达EOS）
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    # 初始化解码变量
    decoded_ids = tf.zeros([batch_size, decode_length], dtype=tf.int64)  # 预分配固定大小
    hit_eos = tf.fill([batch_size], False)   # 全False，没有序列到达EOS
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)  # 起始token
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)  # 初始log概率为0

    def compute_cache_shape_invariants(tensor):
      """计算缓存张量的形状不变量（TPU需要静态形状）。

      对于TPU，tf.while_loop要求指定循环体中各张量的形状约束。
      这里直接使用张量的静态形状作为不变量（TPU上形状是静态的）。
      """
      return tf.TensorShape(tensor.shape.as_list())

    # 使用tf.while_loop执行解码循环
    # tf.while_loop(cond, body, loop_vars, shape_invariants):
    # - cond: 循环条件函数
    # - body: 循环体函数（inner_loop）
    # - loop_vars: 初始循环变量列表
    # - shape_invariants: 各循环变量的形状约束（用于图编译）
    _, _, _, decoded_ids, _, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),           # i: 标量
            tf.TensorShape([batch_size]), # hit_eos: [batch]
            tf.TensorShape([batch_size, 1]),  # next_id: [batch, 1]
            tf.TensorShape([batch_size, decode_length]),  # decoded_ids: 固定大小
            # cache: 嵌套结构，每个张量保持静态形状
            nest.map_structure(compute_cache_shape_invariants, cache),
            tf.TensorShape([batch_size]),  # log_prob: [batch]
        ])
    scores = log_prob  # 最终对数概率作为分数

  return {"outputs": decoded_ids, "scores": scores}


def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                init_cache_fn=_init_transformer_cache,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                sos_id=0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False,
                scope_prefix="body/",
                sampling_temperature=0.0,
                top_k=-1,
                cache=None):
  """给定编码器输出，在CPU/GPU上执行快速解码。

  与TPU版本(fast_decode_tpu)的主要区别：
  - 支持动态形状：decoded_ids从空张量开始，每步用tf.concat追加
  - 形状不变量使用 [None, None] 允许动态增长
  - 不使用inplace_ops（GPU上tf.concat通常足够高效）

  同时实现了贪心解码和束搜索：
  - beam_size=1：贪心解码
  - beam_size>1：束搜索

  Args:
    encoder_output: 编码器输出张量
    encoder_decoder_attention_bias: 编码器-解码器注意力偏置
    symbols_to_logits_fn: 增量解码函数，(ids, step, cache) → (logits, cache)
    hparams: 超参数
    decode_length: 额外解码的时间步数
    vocab_size: 输出词表大小
    init_cache_fn: 初始化缓存的函数
    beam_size: 束宽
    top_beams: 返回最优的前top_beams个序列
    alpha: 长度惩罚系数
    sos_id: 开始符号ID（Start Of Sequence）
    eos_id: 结束符号ID（End Of Sequence）
    batch_size: 批次大小（无输入时必须显式传入）
    force_decode_length: 是否强制解码完整的decode_length步
    scope_prefix: 解码器变量作用域前缀
    sampling_temperature: 采样温度（0=贪心）
    top_k: top-k采样的k值（-1=不限制）
    cache: 已有的额外缓存（如注意力历史，用于可视化）

  Returns:
    解码结果字典 {"outputs": decoded_ids, "scores": scores, "cache": cache}
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  # 初始化解码缓存（CPU/GPU版本：attention_init_length=0，从空缓存开始）
  cache = init_cache_fn(
      cache=cache,
      hparams=hparams,
      batch_size=batch_size,
      attention_init_length=0,    # 初始时缓存中没有历史key/value
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      scope_prefix=scope_prefix)

  if beam_size > 1:  # 执行束搜索
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    decoded_ids, scores, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))  # top_beams=1时启用早停优化

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]  # 取最优beam，去掉sos token
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # 贪心解码

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """贪心解码的单步循环（CPU/GPU版本）。

      与TPU版本的区别：
      - 使用 tf.concat 动态追加新token（不是原地更新）
      - decoded_ids的长度随循环增长

      Args:
        i: 当前步骤编号
        hit_eos: [batch_size]，各序列是否到达EOS
        next_id: [batch_size, 1]，上一步的token
        decoded_ids: [batch_size, current_length]，已生成的序列（动态增长）
        cache: 注意力键值对缓存
        log_prob: [batch_size]，累计对数概率

      Returns:
        更新后的循环变量元组
      """
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = sampling_temperature
      if hparams.sampling_method == "random_per_example":
        next_id = common_layers.sample_temperature_per_example(
            logits, temperature, top_k)
      else:
        if hparams.sampling_method == "argmax":
          temperature = 0.0
        next_id = common_layers.sample_with_temperature(logits, temperature,
                                                        top_k)

      log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
                                  axis=1)
      log_prob += tf.gather_nd(
          log_probs, log_prob_indices) * (1 - tf.to_float(hit_eos))
      # 注意：故意在累加log_prob之后才更新hit_eos
      # 这样会包含第一个EOS的概率，但不包含EOS之后的token
      hit_eos |= tf.equal(next_id, eos_id)

      next_id = tf.expand_dims(next_id, axis=1)
      # CPU/GPU版本：用tf.concat动态追加新token（比TPU的inplace_update灵活）
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)

      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      """判断解码是否应继续。"""
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    # CPU/GPU版本：decoded_ids从空张量开始（不预分配固定大小）
    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)  # 长度为0的空张量
    hit_eos = tf.fill([batch_size], False)
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, cache, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),       # i: 标量
            tf.TensorShape([None]),   # hit_eos: 动态batch维度
            tf.TensorShape([None, None]),  # next_id: 动态形状
            tf.TensorShape([None, None]),  # decoded_ids: 动态增长
            # cache: 用beam_search中的辅助函数处理嵌套缓存的形状
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
            tf.TensorShape([None]),   # log_prob: 动态batch维度
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores, "cache": cache}


@registry.register_model
class TransformerScorer(Transformer):
  """Transformer模型的评分版本，在预测模式下只计算序列得分而不生成序列。

  TransformerScorer与Transformer共享相同的检查点（checkpoint互换）。
  用途：
  - 评估给定序列的概率（如重排序、序列选择）
  - 计算困惑度（perplexity）
  - 对候选序列打分（如在对话系统中选择最佳回复）
  """

  def __init__(self, *args, **kwargs):
    super(TransformerScorer, self).__init__(*args, **kwargs)
    # 将模型名设置为"transformer"，确保与Transformer共享变量名
    self._name = "transformer"
    self._base_name = "transformer"

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """计算目标序列的对数概率得分（而非生成新序列）。

    与标准的infer（生成序列）不同，这里直接对给定的目标序列打分。
    步骤：
    1. 运行完整的模型前向传播，得到所有位置的logits
    2. 计算每个位置的log概率
    3. 提取目标token对应的log概率
    4. 对序列长度求和，得到序列总对数概率

    Args:
      features: 特征字典，必须包含"targets"
      decode_length: 忽略（Scorer不生成新序列）
      beam_size: 忽略
      top_beams: 忽略
      alpha: 忽略
      use_tpu: 忽略

    Returns:
      字典 {"outputs": targets, "scores": sequence_log_probs}
        targets: 目标token ID，形状 [batch_size, target_length]
        scores: 序列对数概率，形状 [batch_size]（越高越好）
    """
    # 忽略所有生成相关的参数
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None

    # 运行完整的前向传播（teacher forcing模式）
    self.hparams.force_full_predict = True
    with tf.variable_scope(self.name):
      logits, _ = self.model_fn(features)
    # logits形状：[batch, time, 1, 1, vocab]（t2t的4D+1格式）
    assert len(logits.shape) == 5
    # 去掉多余维度：[batch, time, 1, 1, vocab] → [batch, time, vocab]
    logits = tf.squeeze(logits, [2, 3])

    # 计算log概率（对logits做log_softmax）
    log_probs = common_layers.log_prob_from_logits(logits)

    targets = features["targets"]
    # targets形状：[batch, time, 1, 1]
    assert len(targets.shape) == 4
    # 去掉多余维度：[batch, time, 1, 1] → [batch, time]
    targets = tf.squeeze(targets, [2, 3])

    # 提取每个时间步目标token的log概率
    # index_last_dim_with_indices: 从log_probs中按targets索引取值
    # 结果形状：[batch, time]
    log_probs = common_layers.index_last_dim_with_indices(log_probs, targets)

    # 对时间维度求和，得到整个序列的对数概率
    # （对数概率相加等价于概率相乘，即序列的联合概率的对数）
    scores = tf.reduce_sum(log_probs, axis=1)

    return {"outputs": targets, "scores": scores}


@registry.register_model
class TransformerEncoder(t2t_model.T2TModel):
  """仅包含编码器的Transformer模型。

  适用于：
  - 文本分类任务（输入序列 → 单一向量/标签）
  - 特征提取（获取序列的编码表示）
  - 可与下游任务结合（如情感分析、命名实体识别等）
  """

  def body(self, features):
    """编码器主体：只进行编码，不解码。

    Args:
      features: 特征字典，包含：
        "inputs": 输入序列，形状 [batch, length, 1, hidden_dim]
        "target_space_id": 目标空间ID

    Returns:
      编码器输出，形状 [batch, length, 1, hidden_dim]
    """
    hparams = self._hparams
    inputs = features["inputs"]
    target_space = features["target_space_id"]

    # 展平输入：4D → 3D
    inputs = common_layers.flatten4d3d(inputs)

    # 准备编码器（计算padding mask、位置编码等）
    (encoder_input, encoder_self_attention_bias, _) = (
        transformer_prepare_encoder(inputs, target_space, hparams))

    # 施加dropout（防止过拟合）
    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    # 执行编码器（多层自注意力 + 前馈网络）
    encoder_output = transformer_encoder(
        encoder_input,
        encoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "inputs"))
    # 添加维度：3D → 4D，符合t2t格式
    encoder_output = tf.expand_dims(encoder_output, 2)

    return encoder_output


@registry.register_model
class TransformerRegressor(TransformerEncoder):
  """用于回归任务的Transformer（继承自TransformerEncoder）。

  在编码器输出上进行平均池化，然后接一个线性层输出标量值。
  最终输出形状为 (batch_size, 1, 1, 1)（单个回归值）。

  适用于：序列级别的回归任务（如句子相似度评分等）
  """

  def top(self, body_output, features):
    """从编码器输出计算单个标量回归值。

    Args:
      body_output: 编码器输出，形状 [batch, length, 1, hidden_dim]
      features: 特征字典（未使用）

    Returns:
      回归预测值，形状 [batch, 1, 1, 1]
    """
    with tf.variable_scope("reg_top_ffn"):
      x = body_output
      # 对长度和空间维度取平均（全局平均池化）
      # axis=[1, 2] 对序列长度和空间维度求平均
      # keepdims=True 保持维度，使结果形状为 [batch, 1, 1, hidden_dim]
      x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
      # 线性层：hidden_dim → 1（单个回归值）
      res = tf.layers.dense(x, 1, name="model_top")
      return res


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  """从特征字典中提取非padding位置的mask。

  对于"packed"数据集（将多个短序列拼接成长序列的格式），
  segmentation字段标记了每个token属于哪个原始序列（0表示padding）。
  通过将segmentation值限制在0~1之间，得到非padding的二值mask。

  Args:
    features: 特征字典
    inputs_or_targets: "inputs"或"targets"，指定是输入还是输出

  Returns:
    非padding mask张量（1=有效位置，0=padding），或None（非packed数据集）
  """
  # 检查segmentation特征是否存在（packed数据集才有）
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    # tf.minimum(x, 1.0)：将大于1的值截断到1（segmentation值>=1表示有效位置）
    # tf.to_float：转换为浮点数
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None


def transformer_prepare_decoder(targets, hparams, features=None, pad=None):
  """为解码器准备输入。

  主要操作：
  1. 计算解码器自注意力偏置（因果mask）
  2. 对于packed数据集，添加跨样本隔离mask
  3. 将目标序列右移一位（teacher forcing：第t步的输入是第t-1步的真实目标）
  4. 添加位置编码

  为什么要右移（shift right）：
  - Transformer解码器是自回归的：预测第t个token时只能看到前t-1个token
  - 右移后，targets[i]的解码器输入是targets[i-1]（真实前一个token）
  - 第0个位置用GO token（pad/零向量）填充，表示序列开始

  Args:
    targets: 目标序列张量，形状 [batch_size, target_length, 1, hidden_dim]
    hparams: 超参数
    features: 可选，完整特征字典（packed数据集需要）
    pad: 可选，右移时用于填充的向量（默认使用零向量）

  Returns:
    (decoder_input, decoder_self_attention_bias) 元组：
      decoder_input: 右移后加了位置编码的解码器输入
      decoder_self_attention_bias: 自注意力偏置（含因果mask）
  """
  if hparams.causal_decoder_self_attention:
    # 因果注意力（标准解码器，每个位置只能看到之前的位置）
    if hparams.prepend_mode == "prepend_inputs_full_attention":
      # prepend模式：输入序列被预置到目标序列前，
      # 输入部分可以全注意力（双向），目标部分使用因果注意力
      decoder_self_attention_bias = (
          common_attention.attention_bias_prepend_inputs_full_attention(
              common_attention.embedding_to_padding(targets)))
    else:
      # 标准因果mask：下三角矩阵，位置i只能attend to位置0..i
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(targets)[1]))
  else:
    # 全注意力（非因果，适用于某些非自回归模型）
    decoder_padding = common_attention.embedding_to_padding(targets)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(decoder_padding))

  if features and "targets_segmentation" in features:
    # "Packed"数据集：同一批次中的不同原始样本不能互相看到
    # 通过attention_bias_same_segment添加额外的mask，
    # 确保只有相同segment_id的位置之间才能注意到对方
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None

  if hparams.proximity_bias:
    # 近邻偏置：鼓励模型关注更近的位置（使用相对位置的正弦函数）
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])

  # 右移目标序列（shift right）：在最左边填充pad，最右边的token被丢弃
  # 这是teacher forcing的关键步骤
  decoder_input = common_layers.shift_right_3d(targets, pad)

  # 添加位置编码到解码器输入
  if hparams.pos == "timing":
    if targets_position is not None:
      # packed数据集：使用给定的位置信息（不是0,1,2...而是原始序列内的位置）
      decoder_input = common_attention.add_timing_signal_1d_given_position(
          decoder_input, targets_position)
    else:
      # 普通数据集：按顺序添加正弦/余弦位置编码
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  elif hparams.pos == "timing_from_features":
    # 从特征中获取位置信号（用于语音等特殊任务）
    decoder_input = common_attention.add_timing_signals_from_features(
        decoder_input, features, hparams.position_features)
  elif hparams.pos == "emb":
    # 可学习的位置嵌入
    decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "targets_positional_embedding",
        targets_position)

  if hparams.activation_dtype == "bfloat16":
    # 如果使用bfloat16精度（TPU常用），将偏置转换为相应类型
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)


def transformer_self_attention_layer(decoder_input,
                                     decoder_self_attention_bias,
                                     layer_idx,
                                     hparams,
                                     encoder_output=None,
                                     encoder_decoder_attention_bias=None,
                                     cache=None,
                                     decode_loop_step=None,
                                     save_weights_to=None,
                                     make_image_summary=False,
                                     layer_collection=None,
                                     recurrent_memory_by_layer=None,
                                     chunk_number=None):
  """单个Transformer解码器的注意力子层。

  包含两个子部分（如果有encoder_output则包含两个注意力子层）：
  1. 解码器自注意力（Self-Attention）：序列内部的注意力
     - 使用因果mask（causal mask），位置i只能attend to位置0..i
     - 快速解码时利用缓存保存历史key/value
  2. 编码器-解码器交叉注意力（Cross-Attention，仅当有encoder_output时）：
     - query来自解码器，key/value来自编码器输出
     - 允许解码器"读取"编码器的输出信息

  每个注意力子层都遵循 "pre-norm residual" 模式：
    output = x + dropout(sublayer(layer_norm(x)))

  Args:
    decoder_input: 解码器输入，形状 [batch, length, hidden_dim]
    decoder_self_attention_bias: 自注意力偏置（含因果mask）
    layer_idx: 层索引（用于命名variable scope）
    hparams: 超参数
    encoder_output: 编码器输出（可选，None表示纯解码器/语言模型）
                   也可以是编码器输出列表（多编码器架构）
    encoder_decoder_attention_bias: 编码器-解码器注意力偏置
    cache: 快速解码的键值对缓存
    decode_loop_step: TPU解码步骤编号
    save_weights_to: 注意力权重存储字典（用于可视化）
    make_image_summary: 是否生成注意力权重的图像摘要
    layer_collection: KFAC优化器的层集合（通常为None）
    recurrent_memory_by_layer: 循环记忆字典（TransformerMemory使用）
    chunk_number: 块编号（循环记忆使用）

  Returns:
    (x, layer_cache) 元组：
      x: 经过注意力子层处理后的输出，形状同decoder_input
      layer_cache: 当前层的缓存（更新后）
  """
  x = decoder_input
  layer = layer_idx
  layer_name = "layer_%d" % layer  # 如 "layer_0", "layer_1", ...
  # 获取当前层的缓存（如果有的话）
  layer_cache = cache[layer_name] if cache is not None else None

  # 获取attention dropout的广播维度
  # 如 "0,1" 表示在batch和heads维度上广播（即同一batch/head中所有位置用相同的mask）
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  # 获取当前层的循环记忆（TransformerMemory/Transformer-XL使用）
  if recurrent_memory_by_layer is not None:
    recurrent_memory = recurrent_memory_by_layer[layer_name]
  else:
    recurrent_memory = None

  # 面积注意力（Area Attention）参数
  # 面积注意力将相邻token组合成"面积"，然后对面积进行注意力计算
  if layer < hparams.get("num_area_layers", 0):
    # 前几层使用面积注意力
    max_area_width = hparams.get("max_area_width", 1)
    max_area_height = hparams.get("max_area_height", 1)
    memory_height = hparams.get("max_area_height", 1)
  else:
    # 其余层使用标准注意力（面积宽高均为1，等价于普通注意力）
    max_area_width = 1
    max_area_height = 1
    memory_height = 1

  with tf.variable_scope(layer_name):
    with tf.variable_scope("self_attention"):
      # ===== 解码器自注意力 =====
      # 先进行layer normalization（pre-norm），然后计算多头自注意力
      y = common_attention.multihead_attention(
          common_layers.layer_preprocess(  # LayerNorm预处理
              x, hparams, layer_collection=layer_collection),
          None,                            # memory=None表示自注意力（Q=K=V）
          decoder_self_attention_bias,     # 因果mask偏置
          hparams.attention_key_channels or hparams.hidden_size,   # key维度
          hparams.attention_value_channels or hparams.hidden_size, # value维度
          hparams.hidden_size,             # 输出维度
          hparams.num_heads,               # 注意力头数
          hparams.attention_dropout,       # 注意力dropout率
          attention_type=hparams.self_attention_type,  # 注意力类型（点积/相对位置等）
          max_relative_position=hparams.max_relative_position,  # 相对位置编码最大距离
          heads_share_relative_embedding=(
              hparams.heads_share_relative_embedding),  # 多头是否共享相对位置编码
          add_relative_to_values=hparams.add_relative_to_values,  # 是否将相对位置加到value
          save_weights_to=save_weights_to,  # 保存注意力权重（可视化用）
          cache=layer_cache,               # 快速解码的缓存
          make_image_summary=make_image_summary,  # 是否生成图像摘要
          dropout_broadcast_dims=attention_dropout_broadcast_dims,
          max_length=hparams.get("max_length"),  # 最大序列长度
          decode_loop_step=decode_loop_step,  # TPU解码步骤
          vars_3d=hparams.get("attention_variables_3d"),  # 是否使用3D变量
          activation_dtype=hparams.get("activation_dtype", "float32"),
          weight_dtype=hparams.get("weight_dtype", "float32"),
          layer_collection=layer_collection,
          recurrent_memory=recurrent_memory,  # 循环记忆
          chunk_number=chunk_number,
          hard_attention_k=hparams.get("hard_attention_k", 0),   # 硬注意力top-k
          gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),  # Gumbel噪声
          max_area_width=max_area_width,   # 面积注意力参数
          max_area_height=max_area_height,
          memory_height=memory_height,
          area_key_mode=hparams.get("area_key_mode", "none"),
          area_value_mode=hparams.get("area_value_mode", "none"),
          training=(hparams.get(                # 是否在训练模式
              "mode",
              tf_estimator.ModeKeys.TRAIN) == tf_estimator.ModeKeys.TRAIN))
      # Post-norm残差连接：x = x + dropout(y)
      x = common_layers.layer_postprocess(x, y, hparams)

    # ===== 编码器-解码器交叉注意力（仅当有encoder_output时） =====
    if encoder_output is not None:
      # 支持多编码器（encoder_output可以是列表）
      if not isinstance(encoder_output, (list,)):
        encoder_output = [encoder_output]
      with tf.variable_scope("encdec_attention"):
        for enc_output in encoder_output:
          # 解码器对编码器输出进行交叉注意力
          # query = 来自解码器（预处理后的x）
          # memory = 来自编码器（enc_output）
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams, layer_collection=layer_collection),
              enc_output,                        # memory: 编码器输出（用于key和value）
              encoder_decoder_attention_bias,    # 编码器padding mask
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              cache=layer_cache,     # 编码器的key/value已预计算缓存
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              layer_collection=layer_collection,
              hard_attention_k=hparams.get("hard_attention_k", 0),
              gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
              max_area_width=max_area_width,
              max_area_height=max_area_height,
              memory_height=memory_height,
              area_key_mode=hparams.get("area_key_mode", "none"),
              area_value_mode=hparams.get("area_value_mode", "none"),
              training=(hparams.get(
                  "mode",
                  tf_estimator.ModeKeys.TRAIN) == tf_estimator.ModeKeys.TRAIN))
          # 残差连接
          x = common_layers.layer_postprocess(x, y, hparams)
  return x, layer_cache


def transformer_decoder_layer(decoder_input,
                              decoder_self_attention_bias,
                              layer_idx,
                              hparams,
                              encoder_output=None,
                              encoder_decoder_attention_bias=None,
                              cache=None,
                              decode_loop_step=None,
                              nonpadding=None,
                              save_weights_to=None,
                              make_image_summary=False,
                              losses=None,
                              layer_collection=None,
                              recurrent_memory_by_layer=None,
                              chunk_number=None):
  """单个完整的Transformer解码器层。

  每个解码器层包含三个子层（如果有encoder_output则三个，否则两个）：
  1. 自注意力（Self-Attention）+ 残差连接
  2. 编码器-解码器交叉注意力（Cross-Attention）+ 残差连接（有encoder_output时）
  3. 前馈网络（Feed-Forward Network，FFN）+ 残差连接

  每个子层都遵循 pre-norm 残差模式：
    output = x + dropout(sublayer(layer_norm(x)))

  Args:
    decoder_input: 解码器输入，形状 [batch, length, hidden_dim]
    decoder_self_attention_bias: 自注意力偏置（含因果mask）
    layer_idx: 层索引
    hparams: 超参数
    encoder_output: 编码器输出（可选）
    encoder_decoder_attention_bias: 编码器-解码器注意力偏置（可选）
    cache: 快速解码缓存
    decode_loop_step: TPU解码步骤
    nonpadding: 非padding位置标记（用于卷积层）
    save_weights_to: 注意力权重存储字典
    make_image_summary: 是否生成注意力图像摘要
    losses: 额外损失列表
    layer_collection: KFAC优化器层集合
    recurrent_memory_by_layer: 循环记忆字典
    chunk_number: 块编号

  Returns:
    x: 经过完整解码器层处理后的输出，形状同decoder_input
  """
  # 步骤1 & 2：执行注意力子层（自注意力 + 可选的交叉注意力）
  x, layer_cache = transformer_self_attention_layer(
      decoder_input=decoder_input,
      decoder_self_attention_bias=decoder_self_attention_bias,
      layer_idx=layer_idx,
      hparams=hparams,
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      cache=cache,
      decode_loop_step=decode_loop_step,
      save_weights_to=save_weights_to,
      make_image_summary=make_image_summary,
      layer_collection=layer_collection,
      recurrent_memory_by_layer=recurrent_memory_by_layer,
      chunk_number=chunk_number)

  layer = layer_idx
  layer_name = "layer_%d" % layer
  with tf.variable_scope(layer_name):
    with tf.variable_scope("ffn"):
      # 步骤3：前馈网络（FFN）
      # FFN结构：Linear(hidden → filter_size) → ReLU → Linear(filter_size → hidden)
      # 其中 filter_size 通常是 hidden_size 的4倍
      y = transformer_ffn_layer(
          common_layers.layer_preprocess(   # 先做LayerNorm
              x, hparams, layer_collection=layer_collection),
          hparams,
          conv_padding="LEFT",     # 左填充（用于卷积型FFN，确保因果性）
          nonpadding_mask=nonpadding,  # 非padding mask（避免在padding位置浪费计算）
          losses=losses,           # 收集FFN的额外损失（如MoE路由损失）
          cache=layer_cache,       # 缓存（某些FFN类型需要）
          decode_loop_step=decode_loop_step,
          layer_collection=layer_collection)
      # 残差连接：x = x + dropout(y)
      x = common_layers.layer_postprocess(x, y, hparams)
      return x


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        layer_collection=None,
                        recurrent_memory_by_layer=None,
                        chunk_number=None):
  """多层Transformer解码器的堆叠。

  将多个transformer_decoder_layer堆叠在一起，构成完整的解码器。
  每层的输出作为下一层的输入（残差连接在每层内部完成）。
  最后还有一个layer normalization（如果使用pre-norm的话）。

  整体架构（每层）：
  ┌─────────────────────────────────┐
  │  输入 x                          │
  │    ↓                            │
  │  LayerNorm → Self-Attention     │
  │    ↓ + Residual                 │
  │  LayerNorm → Cross-Attention    │  （有encoder_output时）
  │    ↓ + Residual                 │
  │  LayerNorm → FFN                │
  │    ↓ + Residual                 │
  │  输出 x                          │
  └─────────────────────────────────┘
  最终输出：LayerNorm(x)

  Args:
    decoder_input: 解码器输入张量，形状 [batch, length, hidden_dim]
    encoder_output: 编码器输出张量（语言模型时为None）
    decoder_self_attention_bias: 解码器自注意力偏置（因果mask）
    encoder_decoder_attention_bias: 编码器-解码器注意力偏置
    hparams: 超参数
    cache: 快速解码的键值对缓存
    decode_loop_step: TPU解码步骤
    name: variable scope名称（默认"decoder"）
    nonpadding: 非padding位置mask（packed数据集使用）
                形状 [batch_size, encoder_length]
    save_weights_to: 注意力权重存储字典
    make_image_summary: 是否生成注意力图像摘要
    losses: 额外损失列表
    layer_collection: KFAC优化器层集合
    recurrent_memory_by_layer: 按层命名的循环记忆字典
    chunk_number: 循环记忆使用的块编号，形状 [batch]

  Returns:
    y: 解码器最终输出，形状 [batch, length, hidden_dim]
  """
  x = decoder_input  # 初始输入

  # 记录MLPerf日志（用于标准化性能测试）
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      },
      hparams=hparams)

  with tf.variable_scope(name):
    # 逐层执行解码器层
    for layer_idx in range(hparams.num_decoder_layers or
                           hparams.num_hidden_layers):
      x = transformer_decoder_layer(
          x,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          encoder_output=encoder_output,
          cache=cache,
          decode_loop_step=decode_loop_step,
          nonpadding=nonpadding,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          losses=losses,
          layer_collection=layer_collection,
          recurrent_memory_by_layer=recurrent_memory_by_layer,
          chunk_number=chunk_number
          )

    # 最后的LayerNorm（如果使用pre-norm，整个堆栈最后需要额外的归一化）
    # 原因：如果在每个子层前做归一化（pre-norm），最后一层的输出还没有被归一化
    # 这一步确保最终输出是经过归一化的
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(
        x, hparams, layer_collection=layer_collection)


@registry.register_model
class TransformerMemory(Transformer):
  """带有跨块循环记忆的Transformer语言模型。

  基于Transformer-XL的思路：
  - 将长序列分割成固定大小的"块"（chunks）
  - 处理每个块时，可以"记住"前面块的信息（通过记忆机制）
  - 记忆的内容被传递给下一块，实现跨块的长程依赖

  与标准Transformer-XL的区别：
  - 支持两种记忆类型：
    * "transformer_xl"：保留最近若干个token的表示
    * "neural_memory"：使用神经记忆网络（类似NTM）

  TODO(kitaev): 考虑重写set_mode以在训练和评估之间切换循环记忆。
  """

  def __init__(self, *args, **kwargs):
    super(TransformerMemory, self).__init__(*args, **kwargs)

    hparams = self._hparams
    # 为每一个解码器层创建对应的循环记忆
    self.recurrent_memory_by_layer = {}
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      if hparams.memory_type == "neural_memory":
        # 神经记忆：使用可寻址的记忆矩阵
        memory = transformer_memory.TransformerMemory(
            batch_size=int(hparams.batch_size / hparams.max_length),
            key_depth=hparams.hidden_size,
            val_depth=hparams.hidden_size,
            memory_size=hparams.split_targets_chunk_length,
            sharpen_factor=1.,
            name=layer_name + "/recurrent_memory")
      elif hparams.memory_type == "transformer_xl":
        # Transformer-XL风格的记忆：保留最近的token表示
        memory = transformer_memory.RecentTokensMemory(
            layer_name + "/recurrent_memory", hparams)
      else:
        raise ValueError("不支持的记忆类型: %s" % hparams.memory_type)
      self.recurrent_memory_by_layer[layer_name] = memory

  @property
  def has_input(self):
    """重写has_input属性，支持无条件生成模式。

    当hparams.unconditional=True时，模型以无条件语言模型方式运行
    （不需要输入序列，纯粹生成）。
    """
    if hasattr(self._hparams, "unconditional") and self._hparams.unconditional:
      return False
    return super(TransformerMemory, self).has_input

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha,
                   use_tpu=False):
    """覆盖束搜索，目前循环记忆只支持慢速版本的束搜索。

    快速束搜索需要对缓存结构进行特殊处理，
    而循环记忆的缓存结构目前不兼容快速束搜索。
    """
    return self._beam_decode_slow(features, decode_length, beam_size,
                                  top_beams, alpha, use_tpu)


# =====================================================================
# 以下是各种超参数配置（HParams）
# 这些函数定义了不同规模和用途的Transformer配置
# =====================================================================

@registry.register_hparams
def transformer_base_v1():
  """Transformer基础超参数集（V1版本）。

  这是原始论文（"Attention Is All You Need"）中的基础配置。
  主要超参数：
  - hidden_size=512: 模型隐藏层维度（每个token的向量大小）
  - num_hidden_layers=6: 编码器和解码器各6层
  - num_heads=8: 注意力头数
  - filter_size=2048: FFN中间层维度（通常是hidden_size的4倍）
  - batch_size=4096: 每批次的token数（而非样本数）
  - label_smoothing=0.1: 标签平滑（防止模型过度自信）
  """
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"           # 使用Layer Normalization
  hparams.hidden_size = 512             # 隐藏层维度（d_model）
  hparams.batch_size = 4096             # 批次大小（token数）
  hparams.max_length = 256              # 最大序列长度
  hparams.clip_grad_norm = 0.           # 梯度裁剪（0表示不裁剪）
  hparams.optimizer_adam_epsilon = 1e-9 # Adam优化器的数值稳定项
  hparams.learning_rate_schedule = "legacy"   # 学习率调度方式
  hparams.learning_rate_decay_scheme = "noam" # Noam学习率衰减（原论文使用）
  hparams.learning_rate = 0.1           # 基础学习率
  hparams.learning_rate_warmup_steps = 4000  # 学习率预热步数
  hparams.initializer_gain = 1.0        # 初始化增益
  hparams.num_hidden_layers = 6         # 编码器和解码器层数（各6层）
  hparams.initializer = "uniform_unit_scaling"  # 参数初始化方式
  hparams.weight_decay = 0.0            # 权重衰减（0=不使用）
  hparams.optimizer_adam_beta1 = 0.9    # Adam的β1参数
  hparams.optimizer_adam_beta2 = 0.98   # Adam的β2参数（原论文值，比默认的0.999小）
  hparams.num_sampled_classes = 0       # 采样softmax的类数（0=全softmax）
  hparams.label_smoothing = 0.1         # 标签平滑系数
  hparams.shared_embedding_and_softmax_weights = True  # 编码器/解码器/softmax共享embedding
  hparams.symbol_modality_num_shards = 16  # embedding分片数（分布式训练用）

  # 以下是Transformer特有的超参数
  hparams.add_hparam("filter_size", 2048)  # FFN中间层维度（d_ff）
  # 如果为0，则使用num_hidden_layers
  hparams.add_hparam("num_encoder_layers", 0)  # 编码器层数（单独设置）
  hparams.add_hparam("num_decoder_layers", 0)  # 解码器层数（单独设置）
  # 注意力相关超参数
  hparams.add_hparam("num_heads", 8)           # 注意力头数（h）
  hparams.add_hparam("attention_key_channels", 0)   # 0=使用hidden_size
  hparams.add_hparam("attention_value_channels", 0) # 0=使用hidden_size
  hparams.add_hparam("ffn_layer", "dense_relu_dense")  # FFN类型
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # 以"dropout"结尾的超参数在非训练模式下自动设为0.0
  hparams.add_hparam("attention_dropout", 0.0)  # 注意力权重的dropout
  hparams.add_hparam("attention_dropout_broadcast_dims", "")
  hparams.add_hparam("relu_dropout", 0.0)       # FFN中ReLU后的dropout
  hparams.add_hparam("relu_dropout_broadcast_dims", "")
  hparams.add_hparam("pos", "timing")    # 位置编码类型：timing（正弦/余弦）、none、emb
  hparams.add_hparam("position_features", "")   # 位置特征（timing_from_features时使用）
  hparams.add_hparam("nbr_decoder_problems", 1) # 解码器任务数
  hparams.add_hparam("proximity_bias", False)   # 是否使用近邻偏置
  hparams.add_hparam("causal_decoder_self_attention", True)  # 是否使用因果注意力
  hparams.add_hparam("use_pad_remover", True)   # 是否移除padding位置（节省计算）
  hparams.add_hparam("self_attention_type", "dot_product")  # 自注意力类型
  hparams.add_hparam("conv_first_kernel", 3)    # 卷积型FFN的第一个卷积核大小
  hparams.add_hparam("attention_variables_3d", False)  # 是否使用3D注意力变量
  hparams.add_hparam("use_target_space_embedding", True)  # 是否使用目标空间嵌入
  # MoE（混合专家）相关超参数（当ffn_layer=="local_moe_tpu"时使用）
  hparams.add_hparam("moe_overhead_train", 1.0)  # 训练时MoE的计算开销倍数
  hparams.add_hparam("moe_overhead_eval", 2.0)   # 评估时MoE的计算开销倍数
  hparams.moe_num_experts = 16       # 专家数量
  hparams.moe_loss_coef = 1e-3       # MoE路由损失系数
  # 评估指标名称覆盖（用于实验对比）
  hparams.add_hparam("overload_eval_metric_name", "")
  # 单向编码器（通过masked attention将编码器变为单向）
  hparams.add_hparam("unidirectional_encoder", False)
  # 硬注意力参数（只选top-k个位置进行注意力）
  hparams.add_hparam("hard_attention_k", 0)
  hparams.add_hparam("gumbel_noise_weight", 0.0)  # Gumbel噪声权重（用于硬注意力训练）
  return hparams


@registry.register_hparams
def transformer_base_v2():
  """Transformer基础超参数集（V2版本）。

  在V1基础上的改进：
  - 使用pre-norm (layer_preprocess_sequence="n")
  - 更高的dropout率，减少过拟合
  - 更长的warmup步数（8000 vs 4000）
  - 更高的基础学习率

  pre-norm vs post-norm：
  - post-norm（V1）：先计算注意力，再做残差连接，最后LayerNorm
    x = LN(x + attention(x))
  - pre-norm（V2）：先LayerNorm，再计算注意力，然后残差连接
    x = x + attention(LN(x))
  pre-norm训练更稳定，但可能影响最终性能
  """
  hparams = transformer_base_v1()
  hparams.layer_preprocess_sequence = "n"     # 在子层前做LayerNorm（pre-norm）
  hparams.layer_postprocess_sequence = "da"   # 子层后做dropout（d）和残差加法（a）
  hparams.layer_prepostprocess_dropout = 0.1  # 残差连接中的dropout
  hparams.attention_dropout = 0.1             # 注意力dropout
  hparams.relu_dropout = 0.1                  # ReLU后的dropout
  hparams.learning_rate_warmup_steps = 8000   # 更长的warmup
  hparams.learning_rate = 0.2
  return hparams


@registry.register_hparams
def transformer_base_vq_ada_32ex_packed():
  """向量量化（VQ）门控的32专家MoE Transformer，用于lm1b packed数据集的TPU训练。"""
  hparams = transformer_base_v2()
  expert_utils.update_hparams_for_vq_gating(hparams)
  hparams.moe_num_experts = 32
  hparams.gating_type = "vq"         # 使用向量量化门控
  # batch_size=5072对应每批256个token、每序列256长度的16个序列
  hparams.batch_size = 5072
  hparams.ffn_layer = "local_moe"    # 局部MoE层
  hparams.shared_embedding_and_softmax_weights = False
  hparams.learning_rate_warmup_steps = 10000
  # lm1b32k_packed的一个epoch约等于27200步（batch_size=128时）
  hparams.learning_rate_decay_steps = 27200
  hparams.num_heads = 4
  hparams.num_blocks = 1
  hparams.moe_k = 1              # 每个token选择1个专家
  hparams.num_decoder_layers = 6
  hparams.label_smoothing = 0.
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_preprocess_sequence = "none"
  hparams.weight_decay = 1e-06
  hparams.attention_dropout = 0.1
  hparams.optimizer = "Adafactor"    # Adafactor优化器（内存效率高）
  hparams.learning_rate_schedule = "linear_warmup*rsqrt_decay*linear_decay"
  hparams.activation_dtype = "float32"
  hparams.learning_rate = 0.1
  hparams.learning_rate_constant = 1.0
  return hparams


@registry.register_hparams
def transformer_topk_16_packed():
  """Top-K门控的16专家MoE Transformer。"""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.gating_type = "topk"   # 使用top-k门控（每步选k个专家）
  hparams.moe_num_experts = 16
  hparams.moe_k = 2              # 每个token选择2个专家
  return hparams


@registry.register_hparams
def transformer_base_vq1_16_nb1_packed_nda_b01_scales():
  """VQ门控，带可学习缩放因子的16专家MoE Transformer。"""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.use_scales = int(True)     # 使用可学习的缩放因子
  hparams.moe_num_experts = 16
  hparams.moe_k = 1
  hparams.beta = 0.1                 # VQ的EMA衰减系数
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.ema = False                # 不使用指数移动平均更新码本
  return hparams


@registry.register_hparams
def transformer_base_vq1_16_nb1_packed_dan_b01_scales():
  """VQ门控，DAN顺序，带可学习缩放因子的16专家MoE Transformer。"""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.use_scales = int(True)
  hparams.moe_num_experts = 16
  hparams.moe_k = 1
  hparams.beta = 0.1
  hparams.ema = False
  return hparams


@registry.register_hparams
def transformer_base_vq1_16_nb1_packed_nda_b01_scales_dialog():
  """用于对话任务的VQ MoE Transformer（更大批次和序列长度）。"""
  hparams = transformer_base_vq1_16_nb1_packed_nda_b01_scales()
  hparams.batch_size = 2048
  hparams.max_length = 1024
  hparams.filter_size = 3072
  return hparams


@registry.register_hparams
def transformer_ada_lmpackedbase():
  """Adafactor优化器，使用标准FFN层（非MoE）的LM packed基础配置。"""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.ffn_layer = "dense_relu_dense"  # 切换回标准FFN
  return hparams


@registry.register_hparams
def transformer_ada_lmpackedbase_dialog():
  """用于对话的Adafactor LM packed基础配置。"""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.max_length = 1024
  hparams.ffn_layer = "dense_relu_dense"
  hparams.batch_size = 4096
  return hparams


@registry.register_hparams
def transformer_ada_lmpackedbase_relative():
  """带相对位置编码的Adafactor LM packed基础配置。"""
  hparams = transformer_base_vq_ada_32ex_packed()
  hparams.ffn_layer = "dense_relu_dense"
  return hparams


@registry.register_hparams
def transformer_base_v3():
  """Transformer基础超参数集（V3版本，当前推荐版本）。

  在V2基础上的改进：
  - 使用更好的Adam参数（beta2=0.997）
  - 使用更清晰的学习率调度公式
  """
  hparams = transformer_base_v2()
  hparams.optimizer_adam_beta2 = 0.997  # 更大的beta2，减慢二阶矩估计的更新
  # 新的学习率调度格式：乘法组合
  # constant * linear_warmup * rsqrt_decay * rsqrt_hidden_size
  # 等价于 Noam 方案（原论文使用的公式）
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


@registry.register_hparams
def transformer_base():
  """Transformer基础模型超参数（默认推荐配置）。

  使用V3版本作为默认基础配置。
  """
  hparams = transformer_base_v3()
  return hparams


@registry.register_hparams
def transformer_big():
  """WMT机器翻译大模型超参数。

  "大"版本与基础版本的主要区别：
  - hidden_size: 512 → 1024
  - filter_size: 2048 → 4096
  - num_heads: 8 → 16
  - dropout: 0.1 → 0.3（更强的正则化）

  注意：batch_size减小到2048是为了适应12GB显存的GPU（如NVIDIA TITAN V）
  """
  hparams = transformer_base()
  hparams.hidden_size = 1024    # 更大的隐藏层
  hparams.filter_size = 4096    # 更大的FFN中间层
  hparams.batch_size = 2048     # 减小批次以适应GPU显存
  hparams.num_heads = 16        # 更多注意力头
  hparams.layer_prepostprocess_dropout = 0.3  # 更强的dropout
  return hparams


@registry.register_hparams
def transformer_tall():
  """用于预训练/微调/混合任务的高层Transformer。

  类BERT/GPT的配置：
  - 12层（更深）
  - hidden_size=768（类BERT base）
  - 12个注意力头
  - 最大长度1024
  - 大词表（65536）
  """
  hparams = transformer_base()
  hparams.batch_size = 2048
  hparams.hidden_size = 768        # BERT-base风格的隐藏层大小
  hparams.filter_size = 3072       # 4 * hidden_size
  hparams.num_hidden_layers = 12   # 更深的网络
  hparams.num_heads = 12           # 注意力头数（hidden_size / num_heads = 64）
  hparams.label_smoothing = 0.0    # 预训练通常不使用标签平滑
  hparams.max_length = 1024        # 更长的上下文窗口
  hparams.eval_drop_long_sequences = True  # 评估时丢弃超长序列
  hparams.multiproblem_mixing_schedule = "pretrain"  # 多任务混合调度
  hparams.multiproblem_vocab_size = 65536  # 大词表
  hparams.clip_grad_norm = 1.0     # 梯度裁剪（防止梯度爆炸）
  return hparams


@registry.register_hparams
def transformer_tall_finetune_tied():
  """CNN/DM摘要任务微调（tied语言模型方式）。

  "tied"指：将摘要任务作为语言模型来微调（输入和输出共享同一序列）
  """
  hparams = transformer_tall()
  hparams.multiproblem_max_input_length = 750   # 最大输入长度
  hparams.multiproblem_max_target_length = 100  # 最大目标长度
  hparams.multiproblem_schedule_max_examples = 0  # 直接进入微调阶段
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_constant = 5e-5  # 微调使用较小学习率
  hparams.learning_rate_warmup_steps = 100
  hparams.learning_rate_decay_steps = 80000
  hparams.multiproblem_target_eval_only = True   # 评估只在目标域
  hparams.multiproblem_reweight_label_loss = True  # 重新加权标签损失
  hparams.multiproblem_label_weight = 1.0
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def transformer_tall_train_tied():
  """CNN/DM摘要任务从头训练（tied语言模型方式）。"""
  hparams = transformer_tall()
  hparams.multiproblem_max_input_length = 750
  hparams.multiproblem_max_target_length = 100
  hparams.multiproblem_schedule_max_examples = 0
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_constant = 2e-4  # 从头训练使用较大学习率
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate_decay_steps = 150000
  hparams.multiproblem_target_eval_only = True
  hparams.multiproblem_reweight_label_loss = True
  hparams.multiproblem_label_weight = 1.0
  hparams.optimizer = "true_adam"
  return hparams


@registry.register_hparams
def transformer_tall_finetune_uniencdec():
  """使用单向编码器和解码器微调CNN/DM摘要。"""
  hparams = transformer_tall()
  hparams.max_input_seq_length = 750
  hparams.max_target_seq_length = 100
  hparams.optimizer = "true_adam"
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_decay_steps = 80000
  hparams.learning_rate_constant = 5e-5
  hparams.learning_rate_warmup_steps = 100
  hparams.unidirectional_encoder = True  # 编码器也使用单向注意力（因果）
  return hparams


@registry.register_hparams
def transformer_tall_train_uniencdec():
  """使用单向编码器和解码器从头训练CNN/DM摘要。"""
  hparams = transformer_tall()
  hparams.max_input_seq_length = 750
  hparams.max_target_seq_length = 100
  hparams.optimizer = "true_adam"
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.learning_rate_decay_steps = 150000
  hparams.learning_rate_constant = 2e-4
  hparams.unidirectional_encoder = True
  return hparams


@registry.register_hparams
def transformer_tall_finetune_textclass():
  """在文本分类任务上微调Transformer tall模型的超参数。"""
  hparams = transformer_tall()
  hparams.learning_rate_constant = 6.25e-5
  hparams.learning_rate_schedule = ("linear_warmup*constant*linear_decay")
  hparams.multiproblem_schedule_max_examples = 0
  hparams.multiproblem_target_eval_only = True
  hparams.learning_rate_warmup_steps = 50
  hparams.learning_rate_decay_steps = 25000
  hparams.multiproblem_reweight_label_loss = True
  hparams.multiproblem_label_weight = 0.95
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm():
  """在64k词表上预训练语言模型的超参数。"""
  hparams = transformer_tall()
  hparams.learning_rate_constant = 2e-4
  hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
  hparams.optimizer = "adam_w"    # AdamW（带权重衰减的Adam）
  hparams.weight_decay = 0.01 * hparams.learning_rate_constant
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.optimizer_adam_epsilon = 1e-8
  hparams.multiproblem_schedule_max_examples = 5e8  # 预训练用大量数据
  hparams.learning_rate_decay_steps = 5000000  # 500万步衰减
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm_tpu_adafactor():
  """TPU上使用Adafactor预训练LM（64k词表）的超参数。"""
  hparams = transformer_tall_pretrain_lm()
  update_hparams_for_tpu(hparams)   # 适配TPU
  hparams.max_length = 1024
  hparams.batch_size = 8            # TPU上的绝对样本数（非token数）
  hparams.multiproblem_vocab_size = 2**16  # 65536词表
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm_tpu_adafactor_large():
  """TPU上预训练大型LM的超参数（1024维隐层）。"""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 32768  # 最大适合16G显存：49152（batch=2时）
  hparams.batch_size = 4
  hparams.multiproblem_mixing_schedule = "constant"  # 恒定混合比例
  # 任务顺序：lm/en-de/en-fr/en-ro/de-en/fr-en/ro-en/cnndm/mnli/squad
  hparams.multiproblem_per_task_threshold = "320,80,160,1,80,160,2,20,10,5"
  return hparams


@registry.register_hparams
def transformer_tall_pretrain_lm_tpu():
  """TPU上使用AdamW预训练LM的超参数。"""
  hparams = transformer_tall_pretrain_lm_tpu_adafactor()
  # update_hparams_for_tpu会重置优化器，所以这里重新设置
  hparams.learning_rate_constant = 2e-4
  hparams.learning_rate_schedule = ("linear_warmup * constant * cosdecay")
  hparams.optimizer = "adam_w"
  hparams.weight_decay = 0.01 * hparams.learning_rate_constant
  return hparams


@registry.register_hparams
def transformer_tall_big():
  """用于LM+MNLI联合训练的更大Transformer（18层）。"""
  hparams = transformer_tall()
  hparams.num_hidden_layers = 18   # 更深的网络
  return hparams


@registry.register_hparams
def transformer_big_single_gpu():
  """单GPU上的大型Transformer超参数（较低dropout，更长warmup）。"""
  hparams = transformer_big()
  hparams.layer_prepostprocess_dropout = 0.1  # 降低dropout（单GPU防止欠拟合）
  hparams.learning_rate_warmup_steps = 16000  # 更长warmup（单GPU更新慢）
  return hparams


@registry.register_hparams
def transformer_base_single_gpu():
  """单GPU上的基础Transformer超参数。"""
  hparams = transformer_base()
  hparams.batch_size = 1024
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def transformer_base_multistep8():
  """模拟8块GPU的多步Adam优化器基础Transformer配置。

  MultistepAdam：在每n步后才更新参数（梯度累积），
  效果等同于使用n倍大的批次训练。
  """
  hparams = transformer_base()
  hparams.optimizer = "multistep_adam"
  hparams.optimizer_multistep_accumulate_steps = 8  # 积累8步梯度再更新
  return hparams


@registry.register_hparams
def transformer_cubbitt():
  """CUBBITT实验中使用的Transformer超参数。

  CUBBITT：Czech-to-English neural machine translation
  使用rsqrt衰减、Adafactor优化器、无dropout、较小批次
  """
  hparams = transformer_big_single_gpu()
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.batch_size = 2900
  hparams.learning_rate_warmup_steps = 8000
  hparams.max_length = 150          # 较短的最大序列长度
  hparams.layer_prepostprocess_dropout = 0  # 无dropout
  hparams.optimizer = "Adafactor"
  return hparams


@registry.register_hparams
def transformer_parsing_base():
  """WSJ句法分析任务的基础Transformer超参数。"""
  hparams = transformer_base()
  hparams.attention_dropout = 0.2   # 更强的注意力dropout（结构化预测任务需要）
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.max_length = 512          # 句子通常比机器翻译长
  hparams.learning_rate_warmup_steps = 16000
  hparams.hidden_size = 1024        # 更大的隐层（解析任务复杂）
  hparams.learning_rate = 0.05
  hparams.shared_embedding_and_softmax_weights = False  # 句法标签不应与词共享embedding
  return hparams


@registry.register_hparams
def transformer_parsing_big():
  """WSJ半监督句法分析的大型Transformer超参数。"""
  hparams = transformer_big()
  hparams.max_length = 512
  hparams.shared_source_target_embedding = False
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.batch_size = 2048
  hparams.learning_rate = 0.05
  return hparams


@registry.register_hparams
def transformer_parsing_ice():
  """冰岛语句法分析和词性标注的Transformer超参数。"""
  hparams = transformer_base_single_gpu()
  hparams.batch_size = 4096
  hparams.shared_embedding_and_softmax_weights = False
  return hparams


@registry.register_hparams
def transformer_tiny():
  """极小的Transformer（用于快速测试和调试）。

  2层，hidden_size=128，适合快速验证代码正确性。
  """
  hparams = transformer_base()
  hparams.num_hidden_layers = 2   # 只有2层
  hparams.hidden_size = 128       # 很小的隐层
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_test():
  """用于单元测试的极小Transformer（比tiny更小）。"""
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 16    # 极小隐层，测试速度快
  hparams.filter_size = 8
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def transformer_small():
  """小型Transformer（训练速度快，常用于实验）。

  适合在单GPU上快速迭代实验。
  """
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256    # 适中的隐层大小
  hparams.filter_size = 1024
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_l2():
  """2层Transformer（与基础配置其他参数相同）。"""
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  return hparams


@registry.register_hparams
def transformer_l4():
  """4层Transformer。"""
  hparams = transformer_base()
  hparams.num_hidden_layers = 4
  return hparams


@registry.register_hparams
def transformer_l8():
  """8层Transformer。"""
  hparams = transformer_base()
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def transformer_l10():
  """10层Transformer。"""
  hparams = transformer_base()
  hparams.num_hidden_layers = 10
  return hparams


@registry.register_hparams
def transformer_h1():
  """单头注意力Transformer（用于消融实验：比较多头vs单头）。"""
  hparams = transformer_base()
  hparams.num_heads = 1
  return hparams


@registry.register_hparams
def transformer_h4():
  """4头注意力Transformer。"""
  hparams = transformer_base()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_h16():
  """16头注意力Transformer。"""
  hparams = transformer_base()
  hparams.num_heads = 16
  return hparams


@registry.register_hparams
def transformer_h32():
  """32头注意力Transformer（消融实验：更多头是否更好）。"""
  hparams = transformer_base()
  hparams.num_heads = 32
  return hparams


@registry.register_hparams
def transformer_k128():
  """key维度为128的Transformer（消融实验）。"""
  hparams = transformer_base()
  hparams.attention_key_channels = 128
  return hparams


@registry.register_hparams
def transformer_k256():
  """key维度为256的Transformer。"""
  hparams = transformer_base()
  hparams.attention_key_channels = 256
  return hparams


@registry.register_hparams
def transformer_ff1024():
  """FFN中间层维度为1024的Transformer（比基础版小一半）。"""
  hparams = transformer_base()
  hparams.filter_size = 1024
  return hparams


@registry.register_hparams
def transformer_ff4096():
  """FFN中间层维度为4096的Transformer（与大模型相同）。"""
  hparams = transformer_base()
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def transformer_dr0():
  """无dropout的Transformer（消融实验：验证dropout的作用）。"""
  hparams = transformer_base()
  hparams.layer_prepostprocess_dropout = 0.0
  return hparams


@registry.register_hparams
def transformer_dr2():
  """更强dropout（0.2）的Transformer。"""
  hparams = transformer_base()
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_ls0():
  """无标签平滑的Transformer（label_smoothing=0）。"""
  hparams = transformer_base()
  hparams.label_smoothing = 0.0
  return hparams


@registry.register_hparams
def transformer_ls2():
  """更强标签平滑（0.2）的Transformer。"""
  hparams = transformer_base()
  hparams.label_smoothing = 0.2
  return hparams


@registry.register_hparams
def transformer_hs256():
  """hidden_size=256的Transformer（比基础版小）。"""
  hparams = transformer_base()
  hparams.hidden_size = 256
  return hparams


@registry.register_hparams
def transformer_hs1024():
  """hidden_size=1024的Transformer（比基础版大）。"""
  hparams = transformer_base()
  hparams.hidden_size = 1024
  return hparams


@registry.register_hparams
def transformer_big_dr1():
  """大型Transformer，dropout=0.1。"""
  hparams = transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_big_enfr():
  """英法翻译任务的大型Transformer（filter_size=8192，更大FFN）。"""
  hparams = transformer_big_dr1()
  hparams.shared_embedding_and_softmax_weights = False  # 英法词表差异大，不共享
  hparams.filter_size = 8192    # 更大的FFN（论文实验设置）
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_big_enfr_tpu():
  """英法翻译大型Transformer的TPU版本。"""
  hparams = transformer_big_enfr()
  # TPU矩阵乘法要求维度是128的倍数，减少头数确保每头维度>=128
  hparams.num_heads = 8
  update_hparams_for_tpu(hparams)
  return hparams


@registry.register_hparams
def transformer_big_dr2():
  """大型Transformer，dropout=0.2（更强正则化）。"""
  hparams = transformer_big_dr1()
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_parameter_attention_a():
  """参数注意力FFN的Transformer（方案A：filter_size=1536）。"""
  hparams = transformer_base()
  hparams.ffn_layer = "parameter_attention"  # 使用参数注意力代替标准FFN
  hparams.filter_size = 1536
  return hparams


@registry.register_hparams
def transformer_parameter_attention_b():
  """参数注意力FFN的Transformer（方案B：更大的key/value通道）。"""
  hparams = transformer_base()
  hparams.ffn_layer = "parameter_attention"
  hparams.filter_size = 512
  hparams.parameter_attention_key_channels = 1024
  hparams.parameter_attention_value_channels = 1024
  hparams.num_heads = 16
  return hparams


@registry.register_hparams
def transformer_prepend_v2():
  """prepend_inputs_masked_attention模式的Transformer（V2）。

  prepend模式：将输入序列拼接到目标序列前，
  模型以语言模型方式处理整个序列。
  """
  hparams = transformer_base_v2()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0  # 不限制最大长度
  return hparams


@registry.register_hparams
def transformer_prepend_v1():
  """prepend_inputs_masked_attention模式的Transformer（V1）。"""
  hparams = transformer_base_v1()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 0
  return hparams


@registry.register_hparams
def transformer_prepend():
  """prepend模式Transformer的默认版本（使用V2）。"""
  return transformer_prepend_v2()


@registry.register_ranged_hparams
def transformer_base_range(rhp):
  """超参数搜索范围（基础Transformer）。

  定义各超参数的搜索空间，用于超参数调优。
  使用随机搜索或贝叶斯优化等方法在这些范围内找最优配置。
  """
  # 学习率在对数尺度上搜索（即搜索数量级）
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  # 从候选值列表中离散选择
  rhp.set_discrete("learning_rate_warmup_steps",
                   [1000, 2000, 4000, 8000, 16000])
  rhp.set_float("initializer_gain", 0.5, 2.0)
  rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
  rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
  rhp.set_float("weight_decay", 0.0, 1e-4)


@registry.register_hparams
def transformer_relative():
  """使用相对位置编码代替绝对位置编码的Transformer。

  相对位置编码（Relative Position Encoding）：
  - 不预先添加位置编码到embedding
  - 在注意力计算中直接考虑两个位置之间的相对距离
  - 对于超过训练长度的序列有更好的泛化能力
  """
  hparams = transformer_base()
  hparams.pos = None                              # 不使用绝对位置编码
  hparams.self_attention_type = "dot_product_relative"  # 使用相对位置注意力
  hparams.max_relative_position = 20             # 最大相对位置距离
  return hparams


@registry.register_hparams
def transformer_relative_tiny():
  """小型相对位置Transformer（用于测试）。"""
  hparams = transformer_relative()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_relative_big():
  """大型相对位置Transformer。"""
  hparams = transformer_big()
  hparams.pos = None
  hparams.self_attention_type = "dot_product_relative"
  hparams.max_relative_position = 20
  return hparams


@registry.register_hparams
def transformer_timeseries():
  """用于时间序列任务的Transformer。

  小批次、快速warmup，适合时间序列预测问题。
  """
  hparams = transformer_small()
  hparams.batch_size = 256      # 时间序列通常样本少，批次小
  hparams.learning_rate_warmup_steps = 2000
  return hparams


@registry.register_hparams
def transformer_mlperf_tpu():
  """MLPerf基准测试的TPU Transformer（2x2 TPU Pod上）。

  MLPerf是机器学习性能的标准基准测试集。
  这个配置遵循参考实现的超参数设置。
  """
  hparams = transformer_base_v3()
  hparams.mlperf_mode = True                    # 启用MLPerf记录模式
  hparams.symbol_modality_num_shards = 1        # TPU不需要多分片
  hparams.max_length = 256                       # packed问题时忽略
  hparams.batch_size = 2048                      # 与参考模型一致的批次
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.attention_dropout_broadcast_dims = "0,1"  # 在batch和heads维度广播dropout
  hparams.relu_dropout_broadcast_dims = "1"         # 在长度维度广播
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"
  return hparams


def update_hparams_for_tpu(hparams):
  """修改超参数以兼容TPU训练。

  TPU（Tensor Processing Unit）训练的特殊需求：
  1. 内存限制：使用Adafactor（比Adam节省内存）
  2. 静态形状：批次大小和序列长度必须固定（会丢弃过长序列，填充过短序列）
  3. 矩阵乘法效率：避免concat操作，不需要多分片embedding

  Args:
    hparams: 超参数对象，将被原地修改

  Returns:
    修改后的hparams（同一对象）
  """
  # Adafactor比Adam使用更少的内存（不需要存储二阶矩估计）
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000

  # 避免TPU上昂贵的concat操作
  # 单分片在多GPU上也有助于更快的参数分发
  hparams.symbol_modality_num_shards = 1

  # TPU不支持自适应批次大小和序列长度
  #