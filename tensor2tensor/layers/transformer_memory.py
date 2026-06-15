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

"""Transformer模型的循环记忆模块（Memory Unit）。

本文件实现了让Transformer能够处理超长序列的"记忆"机制。

背景问题：
  标准Transformer的自注意力是O(n²)的，n是序列长度。
  当处理书籍、长文档等超长文本时，这会导致内存不足。

解决方案——循环记忆（Recurrent Memory）：
  将长序列分割成固定大小的"块"（chunks），
  每次只处理一个块，但通过记忆机制保留前面块的信息。
  这样既能处理超长序列，又避免了O(n²)的内存问题。

本文件提供三种实现：

1. RecurrentMemory（抽象基类）：
   定义记忆接口，实际上是一个无操作（no-op）的占位符。
   所有记忆类必须实现 pre_attention() 和 post_attention()。

2. RecentTokensMemory（Transformer-XL风格）：
   缓存最近若干个token的特征向量，作为下一块的扩展上下文。
   这是论文 "Transformer-XL: Attentive Language Models Beyond
   a Fixed-Length Context" (arXiv:1901.02860) 中描述的方法。
   特点：简单高效，但记忆容量有限（只能看到最近N个token）

3. TransformerMemory（神经图灵机风格）：
   使用可寻址的记忆矩阵，通过内容相似度（余弦相似度）来读写记忆。
   基于论文 "Neural Turing Machines" (arXiv:1410.5401)
   和 "Memory-Efficient Adaptive Computation" (arXiv:1607.00036)。
   特点：更灵活，理论上可以存储更多信息，但实现更复杂

记忆的工作机制：
  pre_attention()：在注意力计算前调用
    → 从记忆中读取相关信息，扩展当前块的上下文
  post_attention()：在注意力和FFN计算后调用
    → 将当前块的信息写入记忆，更新记忆状态
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入通用层工具（形状操作等）
from tensor2tensor.layers import common_layers
# 使用TF 1.x兼容接口
import tensorflow.compat.v1 as tf


class RecurrentMemory(object):
  """循环记忆的抽象基类。

  这个基类定义了记忆模块的接口规范，
  但实际上什么都不做（no-op），相当于"无记忆"。

  子类需要覆盖 pre_attention() 和 post_attention() 方法
  来实现实际的记忆功能。

  接口设计思路：
  - 记忆模块以"插件"方式嵌入Transformer的自注意力层
  - 在注意力计算前后分别调用，不改变自注意力的基本结构
  - 通过扩展 memory_antecedent（注意力的key/value来源）来注入记忆内容

  使用方式（伪代码）：
    # 注意力前
    token, query, memory, bias = recurrent_memory.pre_attention(
        segment, query_antecedent, memory_antecedent, bias)
    # 执行自注意力
    x = multihead_attention(query, memory, bias)
    # 注意力后（更新记忆状态）
    x = recurrent_memory.post_attention(token, x)
  """

  def pre_attention(self, segment, query_antecedent, memory_antecedent, bias):
    """在自注意力计算前调用，用于将记忆信息融入当前计算。

    基类实现：什么都不做，直接返回原始输入（no-op）。

    Args:
      segment: 整数张量，形状 [batch]，表示当前块在序列中的编号。
               用于检测块的切换（segment编号递增表示连续块，
               segment编号减小或重置表示新序列开始）。
      query_antecedent: 查询先验（Query Antecedent），形状 [batch, length_q, channels]
                        即解码器自注意力的输入序列（当前块的特征）。
      memory_antecedent: 记忆先验（Memory Antecedent），通常为None。
                         正常情况下注意力允许传入 [batch, length_m, channels]，
                         但记忆机制目前只支持解码器侧的自注意力（不是cross-attention）。
      bias: 注意力偏置张量（见 attention_bias() 函数），用于屏蔽padding等位置。

    Returns:
      四元组 (data, new_query_antecedent, new_memory_antecedent, new_bias)：
        data: 传递给 post_attention 的状态数据（基类返回None）
        new_query_antecedent: 更新后的查询先验（基类返回原始输入）
        new_memory_antecedent: 更新后的记忆先验（基类返回原始输入）
        new_bias: 更新后的注意力偏置（基类返回原始偏置）
    """
    del segment  # 基类不使用segment参数
    return None, query_antecedent, memory_antecedent, bias

  def post_attention(self, token, x):
    """在自注意力和前馈网络计算后调用，用于更新记忆状态。

    基类实现：什么都不做，直接返回原始输入（no-op）。

    Args:
      token: pre_attention 返回的状态数据（Data），
             可以包含任意需要传递到注意力后处理的状态。
             基类中此值为None。
      x: 自注意力和前馈网络计算完成后的输出张量，
         形状 [batch, length, channels]。

    Returns:
      （可能修改过的）输出张量。基类直接返回原始x。
    """
    assert token is None  # 基类中token应该始终为None
    return x


class RecentTokensMemory(RecurrentMemory):
  """缓存最近几个token特征的记忆模块（Transformer-XL风格）。

  这是 Transformer-XL（https://arxiv.org/abs/1901.02860）中描述的记忆机制。

  工作原理：
  1. 将长序列分割成固定长度的"块"（chunk），例如每块512个token
  2. 处理当前块时，将上一块（或上几块）的隐藏状态拼接到当前块前面
  3. 这样当前块的每个token都能"看到"之前块的token
  4. 处理完当前块后，将当前块的状态保存为"记忆"

  示例（chunk_length=3，tokens_to_cache=3）：
    序列：[T1 T2 T3 | T4 T5 T6 | T7 T8 T9]
    处理块1：注意力窗口=[T1 T2 T3]（无历史记忆）
    处理块2：注意力窗口=[T1 T2 T3 | T4 T5 T6]（T1T2T3是缓存的记忆）
    处理块3：注意力窗口=[T4 T5 T6 | T7 T8 T9]（T4T5T6是新记忆）

  注意事项：
  - 记忆是跨块传递的，存储在非可训练的TF变量中（LOCAL_VARIABLES）
  - 当检测到新序列开始（segment重置为0）时，不从记忆中读取（屏蔽历史）
  - 当前实现假设每次处理完整的一块（不支持每次处理一个token的自回归解码）

  存储的状态变量（非可训练）：
  - previous_segment: 上一块的segment ID，用于检测新序列开始
  - previous_vals: 上一块的特征向量，形状 [batch, tokens_to_cache, hidden_size]
  - previous_bias: 上一块的padding偏置，形状 [batch, 1, 1, tokens_to_cache]
  """

  def __init__(self, name, hparams):
    """初始化RecentTokensMemory。

    Args:
      name: 变量作用域名称，用于标识这个记忆模块（每层有不同的名字）。
      hparams: 超参数对象，使用以下字段：
        - hparams.hidden_size: 隐藏层维度（特征向量大小）
        - hparams.split_targets_chunk_length: 每个块的长度（token数）
        - hparams.num_memory_items（可选）: 要缓存的token数
          如果不设置，默认缓存chunk_length个token（即一整块）
        - hparams.recurrent_memory_batch_size（可选）: 批次中序列数
          如果不设置，从 batch_size / max_length 估算
    """
    hidden_size = hparams.hidden_size
    # 每个块的长度（必须 > 0，才能使用分块记忆）
    self.chunk_length = hparams.split_targets_chunk_length
    assert self.chunk_length > 0, "使用循环记忆时必须启用分块（chunking）"

    # 确定要缓存的token数量
    if hasattr(hparams, "num_memory_items") and hparams.num_memory_items > 0:
      # 显式指定缓存的token数（可以 > chunk_length，表示缓存多个块）
      self.tokens_to_cache = hparams.num_memory_items
    else:
      # 默认：缓存整整一个块（即一个chunk的所有token）
      self.tokens_to_cache = self.chunk_length

    # 估算批次中的序列数
    # TODO(kitaev): 分块代码的实现使得确定每批实际序列数比较麻烦。
    # 数据管道应该在未来某个时间点重新审视。
    if (hasattr(hparams, "recurrent_memory_batch_size")
        and hparams.recurrent_memory_batch_size > 0):
      batch_size_in_sequences = hparams.recurrent_memory_batch_size
    else:
      # 从 batch_size（token数） 和 max_length（序列长度）估算序列数
      batch_size_in_sequences = hparams.batch_size / hparams.max_length

    # 缓存的形状定义
    # memory_shape: 存储上一块的特征向量
    memory_shape = [batch_size_in_sequences, self.tokens_to_cache, hidden_size]
    # bias_shape: 存储上一块的注意力偏置（用于判断是否可以attend to）
    # [batch, 1（头维度）, 1（查询维度）, tokens_to_cache（键维度）]
    bias_shape = [batch_size_in_sequences, 1, 1, self.tokens_to_cache]

    # 创建TF变量（存储跨块传递的状态）
    # 注意：这些变量是LOCAL_VARIABLES（不被保存到checkpoint，不参与训练）
    with tf.variable_scope(name):
      # 上一块的segment ID（用于检测新序列开始）
      self.previous_segment = tf.get_variable(
          "memsegment", (batch_size_in_sequences,),
          dtype=tf.int32, trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],  # 局部变量，不保存
          initializer=tf.constant_initializer(0))      # 初始化为0

      # 上一块的特征向量（要传递给下一块的"记忆"）
      self.previous_vals = tf.get_variable(
          "memvals", memory_shape,
          dtype=tf.float32, trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          initializer=tf.constant_initializer(.0))     # 初始化为零向量

      # 上一块的注意力偏置
      # 初始化为-1e9：表示初始时不能attend to任何记忆（记忆为空）
      self.previous_bias = tf.get_variable(
          "membias", bias_shape,
          dtype=tf.float32, trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          initializer=tf.constant_initializer(-1e9))   # 初始屏蔽所有记忆

  def pre_attention(self, segment, query_antecedent, memory_antecedent, bias):
    """在自注意力前，将缓存的历史token注入当前块的注意力上下文。

    操作：
    1. 如果当前segment为0（新序列开始），额外屏蔽历史记忆
    2. 将历史token (previous_vals) 拼接到当前块的前面，作为新的memory_antecedent
    3. 扩展注意力偏置，包含历史token的偏置
    4. 保存当前块的状态，准备传给 post_attention 更新记忆

    Args:
      segment: 整数张量，形状 [batch]，当前块在序列中的编号（从1开始递增）。
               当segment=0时表示新序列的第一块（不应该attend to历史记忆）。
      query_antecedent: 当前块的特征，形状 [batch, length_q, channels]
      memory_antecedent: 必须为None（只支持自注意力，不支持cross-attention）
      bias: 当前块的注意力偏置（因果mask等）

    Returns:
      四元组 (token, new_query_antecedent, new_memory_antecedent, new_bias)：
        token: (remember_segment, remember_vals, remember_bias) 三元组，
               包含需要在 post_attention 中更新到记忆的数据
        new_query_antecedent: 不变（同输入）
        new_memory_antecedent: 历史记忆 + 当前块拼接后的序列，
                               形状 [batch, tokens_to_cache+length_q, channels]
        new_bias: 扩展后的注意力偏置，包含历史部分
    """
    assert memory_antecedent is None, "循环记忆目前只支持语言模型（无cross-attention）"

    # 处理评估时批次大小可能变化的情况
    memory_batch_size = tf.shape(self.previous_vals)[0]   # 记忆中的批次大小
    current_batch_size = tf.shape(query_antecedent)[0]    # 当前批次大小
    amount_to_pad = memory_batch_size - current_batch_size  # 需要填充的数量

    # ===== 处理新序列开始的情况 =====
    # segment=0 表示新序列的第一块，不应该attend to上一个序列的记忆
    # 当 segment[i]=0 时，对批次中第i个样本的历史记忆添加额外的 -1e9 偏置
    # segment[:, None, None, None] 扩展维度以便广播
    # tf.cast(tf.equal(segment, 0), tf.float32) * -1e9：segment=0的位置加-1e9屏蔽
    previous_bias = self.previous_bias[:current_batch_size, :, :, :] + tf.cast(
        tf.equal(segment[:, None, None, None], 0), tf.float32) * -1e9

    # 取当前批次大小对应的历史特征值
    sliced_previous_vals = self.previous_vals[:current_batch_size, :, :]

    # ===== 构建扩展的注意力上下文 =====
    # new_memory_antecedent = [历史token | 当前块token]
    # stop_gradient：历史记忆的梯度不反传（类似Transformer-XL的做法）
    # 这样记忆只是"读"历史，不会通过时间反向传播梯度（BPTT）
    new_memory_antecedent = tf.concat(
        [tf.stop_gradient(sliced_previous_vals), query_antecedent], 1)

    # ===== 构建扩展的注意力偏置 =====
    # new_bias拼接了历史部分的偏置（previous_bias）和当前块的偏置（bias）
    # 历史部分：tile到当前块的每个查询位置
    # 当前块部分：tile到当前批次大小
    new_bias = tf.concat([
        # 历史token的偏置：[batch, 1, 1, tokens_to_cache] → [batch, 1, chunk_length, tokens_to_cache]
        tf.tile(tf.stop_gradient(previous_bias), [1, 1, self.chunk_length, 1]),
        # 当前块的偏置：[1, 1, chunk_length, chunk_length] → [batch, 1, chunk_length, chunk_length]
        tf.tile(bias, [current_batch_size, 1, 1, 1]),
    ], -1)  # 在最后一个维度（key的位置维度）拼接

    # ===== 准备要更新到记忆的数据（传给post_attention）=====
    # remember_segment：当前的segment ID（填充到memory_batch_size）
    remember_segment = tf.pad(segment, [[0, amount_to_pad]])

    # remember_vals：下一块的"记忆"就是当前块的特征
    # TODO(kitaev): 代码假设segment编号要么递增要么重置为0。
    # 这个假设在自回归逐token解码时不成立。
    remember_vals = tf.pad(query_antecedent,
                           [[0, amount_to_pad], [0, 0], [0, 0]])

    # remember_bias：下一块的记忆偏置
    # 当前块中至少有一个查询位置能attend to的token，其bias取max（非padding）
    # tf.reduce_max(bias, -2): 对查询维度取max，保留最宽松的（非-1e9）偏置
    # 这样只要token不是padding，它就会被记忆
    remember_bias = tf.tile(
        tf.reduce_max(bias, -2, keepdims=True),  # [1, 1, 1, chunk_length]
        [memory_batch_size, 1, 1, 1])            # → [memory_batch, 1, 1, chunk_length]

    # 如果需要缓存多个块（tokens_to_cache > chunk_length）
    if self.chunk_length < self.tokens_to_cache:
      # 将历史记忆和当前块拼接，保留最新的tokens_to_cache个token
      remember_vals = tf.concat([self.previous_vals, remember_vals], 1)
      remember_bias = tf.concat([
          # 历史偏置：已经是新序列的屏蔽（segment=0时添加-1e9）
          self.previous_bias - 1e9 * tf.cast(
              tf.equal(
                  tf.pad(segment, [[0, amount_to_pad]])[:, None, None, None],
                  0), tf.float32),
          remember_bias
      ], -1)

    # 只保留最新的tokens_to_cache个token
    if self.chunk_length != self.tokens_to_cache:
      remember_vals = remember_vals[:, -self.tokens_to_cache:, :]  # 取最后N个
      remember_bias = remember_bias[:, :, :, -self.tokens_to_cache:]  # 取最后N个偏置

    # token包含需要传递给post_attention的所有状态
    token = (remember_segment, remember_vals, remember_bias)

    return token, query_antecedent, new_memory_antecedent, new_bias

  def post_attention(self, token, x):
    """在自注意力和前馈网络后，将当前块的状态保存到记忆变量中。

    使用TF的 assign 操作更新非可训练变量，
    通过 control_dependencies 确保在返回x之前更新已完成。

    Args:
      token: pre_attention 返回的三元组 (remember_segment, remember_vals, remember_bias)
             - remember_segment: 要保存的segment ID
             - remember_vals: 要保存的特征向量（当前块的特征）
             - remember_bias: 要保存的注意力偏置
      x: 当前块经过自注意力和FFN后的输出，形状 [batch, length, channels]

    Returns:
      x的一个恒等变换（tf.identity），但带有控制依赖确保记忆更新完成后才返回。
    """
    # control_dependencies: 确保在执行x之前，先执行所有的assign操作
    with tf.control_dependencies([
        self.previous_segment.assign(token[0]),  # 更新segment ID
        self.previous_vals.assign(token[1]),     # 更新特征缓存
        self.previous_bias.assign(token[2]),     # 更新偏置缓存
        ]):
      # tf.identity: 等效于直接返回x，但确保控制依赖被触发
      return tf.identity(x)


class TransformerMemory(object):
  """基于神经图灵机（NTM）的可寻址记忆模块。

  基于论文：
  - "Neural Turing Machines" (arXiv:1410.5401)
  - "Memory-Efficient Adaptive Computation" (arXiv:1607.00036)

  与RecentTokensMemory的区别：
  - RecentTokensMemory：直接缓存最近的特征向量（简单高效）
  - TransformerMemory：使用可寻址记忆矩阵，通过内容相似度读写（更灵活）

  工作机制：
  1. 读操作（Read）：
     - 将输入x转换为query向量
     - 计算query与记忆中每个slot的余弦相似度（内容寻址）
     - 用softmax权重对记忆值加权求和，得到检索结果
  2. 写操作（Write）：
     - 根据读操作的权重和最少使用策略，决定写入哪个slot
     - 先"擦除"（erase），再"添加"（add）新内容

  记忆状态变量（非可训练的TF变量）：
  - mem_vals: 记忆值矩阵，形状 [batch, memory_size, val_depth]
  - mean_logits: 每个记忆slot的平均被访问logits（用于"最少使用"策略）
  - segment_number: 当前处理的序列编号（检测新序列并重置记忆）
  """

  def __init__(self, batch_size, key_depth, val_depth, memory_size,
               sharpen_factor=1., name="neural_memory"):
    """初始化神经记忆模块。

    Args:
      batch_size: 批次大小（批次中的序列数）。
                  注意：这是固定的，与运行时实际批次大小不一定相同。
      key_depth: 记忆key的维度（寻址用，影响相似度计算的精度）。
      val_depth: 记忆value的维度（存储内容的维度，通常等于hidden_size）。
      memory_size: 记忆槽（memory slots）的数量，即记忆矩阵的"行数"。
                   更多的槽意味着更大的记忆容量，但也更慢。
      sharpen_factor: 注意力锐化系数，默认1.0。
                      更大的值使注意力更集中（更接近one-hot），
                      更小的值使注意力更分散（近似均匀读取）。
      name: 变量作用域名称（默认"neural_memory"）。
    """
    self.name = name
    self.batch_size = batch_size
    self.key_depth = key_depth
    self.val_depth = val_depth
    self.memory_size = memory_size
    self.sharpen_factor = sharpen_factor

    with tf.variable_scope(name):
      # segment_number：记录当前处理到哪个序列（用于检测新序列并重置记忆）
      # 初始化为极大值100000，确保第一次处理时会触发重置
      self.segment_number = tf.get_variable(
          "segment_number", [self.batch_size],
          dtype=tf.int32, trainable=False,
          initializer=tf.constant_initializer(100000))

      # mem_vals：记忆值矩阵（存储信息的地方）
      # 形状：[batch_size, memory_size, val_depth]
      # 初始化为零（空记忆）
      self.mem_vals = tf.get_variable(
          "memvals", [self.batch_size, self.memory_size, self.val_depth],
          dtype=tf.float32, trainable=False,
          initializer=tf.constant_initializer(.0))

      # mean_logits：每个记忆slot的平均访问logits（"最少使用"指标）
      # 用于写操作时决定覆盖哪个slot（倾向于覆盖最少被使用的slot）
      self.mean_logits = tf.get_variable(
          "meanlogits", [self.batch_size, self.memory_size],
          dtype=tf.float32, trainable=False,
          initializer=tf.constant_initializer(.0))

  def _norm(self, x):
    """计算向量的L2范数（数值安全版本）。

    添加1e-7避免除零错误（当向量接近零向量时）。

    Args:
      x: 输入张量，最后一维是特征维度

    Returns:
      各向量的L2范数，形状比输入少最后一个维度（keepdims=True保留维度）
    """
    # tf.reduce_sum(tf.square(x), axis=-1) 计算各元素平方和（沿特征维度）
    # +1e-7 避免开方时出现精度问题（特别是梯度计算时的数值不稳定）
    return tf.sqrt(tf.reduce_sum(tf.square(x), keepdims=True, axis=-1) + 1e-7)

  def _address_content(self, x):
    """基于内容相似度寻址记忆（Content-Based Addressing）。

    原理：
    1. 将输入x和记忆值mem_vals分别通过线性层映射到key空间
    2. 计算输入query和记忆key之间的余弦相似度
    3. 乘以sharpen_factor（锐化系数）得到最终的寻址logits

    余弦相似度 = (a·b) / (|a||b|)
    取值范围[-1, 1]，值越大表示越相似。

    Args:
      x: 输入张量，形状 [batch_size, length, depth]

    Returns:
      寻址logits，形状 [batch_size, length, memory_size]
      每个位置与每个记忆slot的相似度分数。
    """
    # 将记忆值转换为key（用于相似度比较）
    # mem_vals: [batch, memory_size, val_depth]
    # mem_keys: [batch, memory_size, key_depth]
    mem_keys = tf.layers.dense(self.mem_vals, self.key_depth,
                               bias_initializer=tf.constant_initializer(1.0),
                               name="mem_key")

    # 将输入x转换为query（与记忆key在同一空间中比较）
    # mem_query: [batch, length, key_depth]
    mem_query = tf.layers.dense(x, self.key_depth,
                                bias_initializer=tf.constant_initializer(1.0),
                                name="mem_query")

    # 计算范数（用于归一化，得到余弦相似度）
    # _norm(mem_query): [batch, length, 1]
    # _norm(mem_keys):  [batch, memory_size, 1]
    # matmul的结果: [batch, length, memory_size]
    norm = tf.matmul(self._norm(mem_query), self._norm(mem_keys),
                     transpose_b=True)  # transpose_b: mem_keys的最后两维转置

    # 计算点积（未归一化的相似度）
    # [batch, length, key_depth] × [batch, key_depth, memory_size]
    # = [batch, length, memory_size]
    dot_product = tf.matmul(mem_query, mem_keys, transpose_b=True)

    # 余弦相似度 = 点积 / 范数乘积
    cos_dist = tf.div(dot_product, norm + 1e-7, name="cos_dist")

    # 乘以锐化系数（更大的sharpen_factor使注意力更集中）
    access_logits = self.sharpen_factor * cos_dist
    return access_logits

  def read(self, x):
    """从记忆中读取信息。

    外部组件可以通过一个简单的MLP来使用读取结果：
    例如：output = fn(x @ W_x + retrieved_mem @ W_m)

    操作流程：
    1. 调用 _address_content 计算x与各记忆slot的相似度logits
    2. softmax得到读取权重（所有slot的权重和为1）
    3. 用权重对记忆值加权求和，得到检索结果

    Args:
      x: 查询张量，形状 [batch_size, length, depth]

    Returns:
      二元组 (access_logits, retrieved_mem)：
        access_logits: 记忆寻址logits，形状 [batch_size, length, memory_size]
                       （也用于后续的写操作）
        retrieved_mem: 检索到的记忆内容，形状 [batch_size, length, val_depth]
                       是所有记忆slot的加权和
    """
    # 计算内容寻址logits
    access_logits = self._address_content(x)
    # softmax归一化，得到各slot的读取权重（和为1）
    weights = tf.nn.softmax(access_logits)  # [batch, length, memory_size]

    # 加权求和：每个查询位置的检索结果 = Σ weights[i] * mem_vals[i]
    # expand_dims后：
    #   weights: [batch, length, memory_size, 1]
    #   mem_vals: [batch, 1, memory_size, val_depth]
    # 相乘后：[batch, length, memory_size, val_depth]
    # reduce_sum(axis=2)：对memory_size维度求和 → [batch, length, val_depth]
    retrieved_mem = tf.reduce_sum(
        tf.multiply(tf.expand_dims(weights, 3),
                    tf.expand_dims(self.mem_vals, axis=1)), axis=2)
    return access_logits, retrieved_mem

  def write(self, x, access_logits):
    """向记忆中写入信息。

    写入策略结合了内容相似度和最少使用（Least Recently Used, LRU）：
    - 倾向于写入与当前输入相似的slot（内容寻址）
    - 减去之前的平均logits（偏向于写入最少使用的slot）

    写入操作（类NTM的擦除-添加机制）：
    1. 擦除（Erase）：对要写入的slot，清除旧的内容（乘以1-erase_gate）
    2. 添加（Add）：写入新的候选值（乘以write_weight）

    最终记忆更新：mem = mean(mem * erase_mask + write_content)

    基于论文：arXiv:1607.00036v2

    Args:
      x: 输入张量，形状 [batch_size, length, depth]
      access_logits: read()返回的寻址logits，形状 [batch_size, length, memory_size]

    Returns:
      写操作的TF op（执行后更新mem_vals和mean_logits变量）
    """
    # gamma：门控参数，控制"最少使用"策略的强度
    # gamma ∈ (0, 1)：gamma接近1时完全使用LRU，接近0时完全依赖内容寻址
    gamma = tf.layers.dense(x, 1, activation=tf.sigmoid, name="gamma")

    # 写入logits = 内容logits - gamma × 平均logits（最少使用调整）
    # 这样使得最近很少被使用的slot（mean_logits小）更容易被写入
    write_logits = access_logits - gamma * tf.expand_dims(self.mean_logits, 1)

    # 候选写入值：通过ReLU线性层将输入转换为要写入的内容
    candidate_value = tf.layers.dense(x, self.val_depth,
                                      activation=tf.nn.relu,
                                      name="candidate_value")

    # 擦除门：决定每个slot要被擦除多少（sigmoid ∈ (0,1)）
    erase_gates = tf.layers.dense(x, self.memory_size,
                                  activation=tf.nn.sigmoid,
                                  name="erase")

    # softmax写入权重（决定往哪个slot写入）
    write_weights = tf.nn.softmax(write_logits)  # [batch, length, memory_size]

    # 擦除权重：1 - erase_gate × write_weight（要保留的比例）
    # 写入权重大 + 擦除门大 → 大幅清除旧内容
    # 形状：[batch, length, memory_size, 1]（扩展以便广播）
    erase_weights = tf.expand_dims(1 - erase_gates * write_weights, 3)

    # 擦除操作：记忆乘以保留系数（接近0则清除）
    # mem_vals: [batch, 1, memory_size, val_depth]（扩展length维度）
    # erase_weights: [batch, length, memory_size, 1]
    # 结果：[batch, length, memory_size, val_depth]
    erase = tf.multiply(erase_weights,
                        tf.expand_dims(self.mem_vals, 1))

    # 添加操作：写入新内容
    # write_weights: [batch, length, memory_size, 1]
    # candidate_value: [batch, length, 1, val_depth]
    # 结果：[batch, length, memory_size, val_depth]
    addition = tf.multiply(
        tf.expand_dims(write_weights, 3),
        tf.expand_dims(candidate_value, 2))

    # 更新记忆：取所有length位置的平均，更新mem_vals
    # reduce_mean(axis=1)：对序列长度维度求平均
    update_value_op = self.mem_vals.assign(
        tf.reduce_mean(erase + addition, axis=1))

    # 更新mean_logits（滑动平均，系数0.1旧值，0.9新值）
    # 这使得mean_logits追踪每个slot的平均访问频率
    with tf.control_dependencies([update_value_op]):
      write_op = self.mean_logits.assign(
          self.mean_logits * 0.1 + tf.reduce_mean(write_logits * 0.9, axis=1))
      return write_op

  def set(self, mem_vals, mean_logits):
    """直接设置记忆值和平均logits（用于恢复记忆状态）。

    Args:
      mem_vals: 要设置的记忆值矩阵，形状 [batch, memory_size, val_depth]
      mean_logits: 要设置的平均logits，形状 [batch, memory_size]

    Returns:
      分组操作（tf.group），执行后同时更新两个变量
    """
    set_op = tf.group([
        self.mem_vals.assign(mem_vals),
        self.mean_logits.assign(mean_logits)])
    return set_op

  def get(self):
    """获取当前的记忆值和平均logits（用于保存记忆状态）。

    Returns:
      (mem_vals, mean_logits) 二元组
    """
    return self.mem_vals, self.mean_logits

  def update_segment_number(self, segment_number):
    """更新记录的序列编号。

    Args:
      segment_number: 新的序列编号张量，形状 [batch]

    Returns:
      assign操作
    """
    return self.segment_number.assign(segment_number)

  def reset(self, entries_to_reset):
    """重置指定批次位置的记忆（用于新序列开始时清空记忆）。

    当检测到某些批次位置开始了新序列（segment_number减小），
    需要清空这些位置的记忆，避免旧序列的信息污染新序列。

    Args:
      entries_to_reset: 1D整数张量，包含需要重置的批次索引。
                        例如 [0, 2] 表示重置第0和第2个序列的记忆。

    Returns:
      分组的reset操作（同时重置mem_vals和mean_logits）
    """
    # 需要重置的序列数量
    num_updates = tf.size(entries_to_reset)

    # scatter_update: 在entries_to_reset指定的索引位置写入全零
    # tf.tile: 创建 num_updates 份全零矩阵
    update_vals = tf.scatter_update(
        self.mem_vals, entries_to_reset,
        tf.tile(tf.expand_dims(
            tf.fill([self.memory_size, self.val_depth], .0), 0),  # [1, memory_size, val_depth]
                [num_updates, 1, 1]))  # → [num_updates, memory_size, val_depth]

    update_logits = tf.scatter_update(
        self.mean_logits, entries_to_reset,
        tf.tile(tf.expand_dims(
            tf.fill([self.memory_size], .0), 0),  # [1, memory_size]
                [num_updates, 1]))  # → [num_updates, memory_size]

    reset_op = tf.group([update_vals, update_logits])
    return reset_op

  def pre_attention(self, segment_number, query_antecedent,
                    memory_antecedent, bias):
    """在自注意力前，从神经记忆中读取相关信息。

    操作流程：
    1. 检测新序列（segment_number减小）→ 重置对应的记忆
    2. 更新segment_number变量
    3. 从记忆中读取与当前输入相关的内容
    4. 将读取结果保存到memory_results中，传给post_attention使用

    注意：这里的记忆读取结果不直接加入注意力上下文（与RecentTokensMemory不同）。
    而是在 post_attention 中，通过MLP将读取结果融入FFN后的输出。

    Args:
      segment_number: 整数张量，形状 [batch]，当前块的序列编号。
      query_antecedent: 当前块的特征，形状 [batch, length_q, channels]
      memory_antecedent: 必须为None（只支持语言模型的自注意力）
      bias: 注意力偏置

    Returns:
      四元组 (memory_results, query_antecedent, memory_antecedent, bias)：
        memory_results: 字典，包含记忆读取的中间结果：
                        {"x": padded_x, "access_logits": ..., "retrieved_mem": ...}
        其余三个与输入相同（不修改注意力的直接输入）
    """
    with tf.variable_scope(self.name + "/pre_attention", reuse=tf.AUTO_REUSE):
      assert memory_antecedent is None, "只支持语言模型（无cross-attention记忆）"

      # 确保segment_number的大小不超过batch_size
      with tf.control_dependencies([
          tf.assert_greater_equal(self.batch_size, tf.size(segment_number))]):
        # 计算需要填充的数量（当实际批次比self.batch_size小时）
        difference = self.batch_size - tf.size(segment_number)
        # 将segment_number填充到self.batch_size大小
        segment_number = tf.pad(segment_number, [[0, difference]])

        # 检测哪些批次位置的序列编号减小了（即开始了新序列）
        # tf.less(segment_number, self.segment_number)：当前编号 < 历史编号 → 新序列
        # tf.where：找出这些位置的索引
        reset_op = self.reset(tf.reshape(tf.where(
            tf.less(segment_number, self.segment_number)), [-1]))

      memory_results = {}
      with tf.control_dependencies([reset_op]):
        # 更新segment_number后，从记忆中读取
        with tf.control_dependencies([
            self.update_segment_number(segment_number)]):
          # 将输入填充到self.batch_size大小（与记忆变量大小匹配）
          x = tf.pad(query_antecedent, [
              [0, difference], [0, 0], [0, 0]])
          # 从记忆中读取（内容寻址 + 加权求和）
          access_logits, retrieved_mem = self.read(x)

      # 保存读取结果，传给post_attention使用
      memory_results["x"] = x                          # 填充后的输入
      memory_results["access_logits"] = access_logits  # 寻址logits（写操作需要）
      memory_results["retrieved_mem"] = retrieved_mem  # 检索到的记忆内容

      # 不修改注意力的直接输入（与RecentTokensMemory不同，这里不扩展上下文）
      return memory_results, query_antecedent, memory_antecedent, bias

  def post_attention(self, token, x):
    """在自注意力和前馈网络后，用读取的记忆内容增强输出，并更新记忆。

    操作：
    1. 将注意力输出 x 与记忆读取结果 retrieved_mem 通过MLP融合：
       output = x @ W_x + retrieved_mem @ W_m
    2. 将当前输入和访问logits写入记忆（更新记忆状态）

    这里记忆的读取结果以残差方式加入到注意力输出上，
    让模型可以利用记忆中存储的历史信息。

    Args:
      token: pre_attention 返回的 memory_results 字典，包含：
             - "x": 填充后的输入（写操作需要）
             - "access_logits": 寻址logits（写操作需要）
             - "retrieved_mem": 检索到的记忆内容
      x: 自注意力和FFN后的输出，形状 [batch, length, channels]

    Returns:
      融合了记忆信息的增强输出，形状同x
    """
    with tf.variable_scope(self.name + "/post_attention", reuse=tf.AUTO_REUSE):
      depth = common_layers.shape_list(x)[-1]            # 特征维度
      actual_batch_size = common_layers.shape_list(x)[0] # 实际批次大小（可能 < self.batch_size）

      # 取对应批次的记忆读取结果
      # tf.range(actual_batch_size)：只取前actual_batch_size个（去掉填充的部分）
      memory_output = tf.gather(token["retrieved_mem"],
                                tf.range(actual_batch_size))

      # 融合注意力输出和记忆读取结果
      # x @ W_x（线性变换，不带bias）+ retrieved_mem @ W_m（线性变换，带bias）
      # 这是一个简单的MLP，将两个信息源融合
      output = tf.add(tf.layers.dense(x, depth, use_bias=False),   # 注意力输出的变换
                      tf.layers.dense(memory_output, depth))         # 记忆内容的变换

      # 确保在返回output前，完成记忆的写入更新
      with tf.control_dependencies([output]):
        with tf.control_dependencies([
            # 将当前输入和访问logits写入记忆（更新mem_vals和mean_logits）
            self.write(token["x"], token["access_logits"])]):
          return tf.identity(output)
