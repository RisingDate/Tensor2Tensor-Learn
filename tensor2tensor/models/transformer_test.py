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

"""Transformer模型的单元测试文件。

本文件包含对 transformer.py 中各核心功能的测试用例，主要测试以下内容：
1. 基础Transformer前向传播（训练模式下的输出形状验证）
2. 语音识别（LibriSpeech）任务的Transformer变体
3. 慢速推理与快速推理（缓存优化）的结果一致性
4. 束搜索解码（Beam Decode）的正确性
5. TPU和非TPU推理结果的一致性
6. 监督注意力损失（Encoder-Decoder Attention Loss）的计算
7. TransformerScorer变体的打分功能

测试框架说明：
- 使用 TensorFlow 的 tf.test.TestCase 作为基类
- 测试基于随机生成的输入数据（非真实数据）
- 通过比较输出形状和数值来验证正确性

关键概念补充：
- 慢速推理（Slow Inference）：每次生成一个token，没有缓存，速度慢但简单
- 快速推理（Fast Inference）：使用KV缓存，避免重复计算历史token的注意力，速度快
- TPU（Tensor Processing Unit）：谷歌专为神经网络加速设计的硬件芯片
- 束搜索（Beam Search）：每步保留top-k个候选序列，平衡探索与效率
- Scorer：不生成序列，而是对已有序列计算概率得分的模型变体
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# numpy：科学计算库，提供多维数组和数学函数
# 这里用来生成随机测试输入数据
import numpy as np

# 导入LibriSpeech数据生成器（语音识别数据集，用于测试语音输入的Transformer）
from tensor2tensor.data_generators import librispeech
# 导入问题超参数（problem_hparams），用于构造测试时的问题配置
from tensor2tensor.data_generators import problem_hparams
# 导入被测试的transformer模块（包含Transformer类和各种超参数配置）
from tensor2tensor.models import transformer

# 导入TensorFlow 1.x兼容接口
import tensorflow.compat.v1 as tf
# 导入estimator，用于获取训练/推理/评估模式的枚举值
from tensorflow.compat.v1 import estimator as tf_estimator


# ============================================================
# 测试用全局常量
# 这些常量定义了测试时使用的数据规模，
# 设置较小的值是为了让测试快速运行
# ============================================================

# 批次大小：每批处理3个样本
BATCH_SIZE = 3

# 输入序列长度：5个token
INPUT_LENGTH = 5

# 目标序列长度（解码器输出长度）：7个token
TARGET_LENGTH = 7

# 词表大小：10个词（测试用，实际任务通常有数万词）
VOCAB_SIZE = 10


def get_model(hparams=None, mode=tf_estimator.ModeKeys.TRAIN,
              has_input=True, model_cls=transformer.Transformer):
  """构建一个用于测试的Transformer模型及其输入特征。

  该函数用于在测试中方便地创建一个小型Transformer模型和随机测试数据。

  整个创建流程：
  1. 获取或创建超参数对象（hparams），将其设置为极小规模以加速测试
  2. 创建问题超参数（p_hparams），描述输入/输出词表大小等
  3. 生成随机的输入和目标序列（整数token ID）
  4. 将数据组装为 features 字典（Transformer接受的格式）
  5. 实例化并返回模型对象和features字典

  Args:
    hparams: 超参数对象。如果为None，使用 transformer_tiny() 中的默认配置。
             hparams 控制模型大小，如 hidden_size（隐藏层维度）、
             num_heads（多头注意力的头数）等。
    mode: 运行模式，来自 tf_estimator.ModeKeys：
          - TRAIN：训练模式（启用dropout等正则化）
          - EVAL：评估模式
          - PREDICT：推理/预测模式（用于生成序列）
    has_input: 是否有输入序列（True=seq2seq任务，False=语言模型任务）。
               语言模型任务没有独立的编码器输入，只有目标序列。
    model_cls: 要实例化的模型类，默认是 transformer.Transformer。
               也可以是其他变体如 TransformerScorer。

  Returns:
    元组 (model, features)：
      model: 实例化的Transformer模型对象
      features: 包含输入/目标数据的字典，格式如下：
                {
                  "inputs": tf.Tensor,         # 输入token ID（has_input=True时存在）
                  "targets": tf.Tensor,        # 目标token ID
                  "target_space_id": tf.Tensor # 目标空间ID（通常表示语言/任务类型）
                }
  """
  # 如果没有提供超参数，使用 transformer_tiny() 的默认配置
  # transformer_tiny() 是一个极小的配置，适合测试和快速实验
  if hparams is None:
    hparams = transformer.transformer_tiny()

  # 强制将模型尺寸设为极小值以加速测试
  hparams.hidden_size = 8      # 隐藏层维度：8（正常模型通常是512或更大）
  hparams.filter_size = 32     # 前馈网络的中间维度：32（正常是2048）
  hparams.num_heads = 1        # 注意力头数：1（正常是8或16）
  hparams.layer_prepostprocess_dropout = 0.0  # 关闭dropout（测试时需要确定性结果）

  # 如果还没有设置问题超参数，创建一个测试用的问题超参数
  # problem_hparams.test_problem_hparams 创建一个简单的输入-目标映射配置
  if hparams.get("problem_hparams", None) is None:
    p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE,  # 输入词表大小
                                                     VOCAB_SIZE,  # 目标词表大小
                                                     hparams)

  # 如果测试不需要输入（如语言模型），从问题配置中删除输入模态
  # 这样模型就不会尝试处理不存在的输入
  if not has_input:
    del p_hparams.modality["inputs"]

  # 将问题超参数设置到主超参数中
  hparams.problem_hparams = p_hparams

  # 生成随机输入数据（整数token ID）
  # np.random.randint(VOCAB_SIZE, size=(...)) 生成 [0, VOCAB_SIZE) 范围内的随机整数
  # 形状: [BATCH_SIZE, INPUT_LENGTH, 1, 1]
  #   - BATCH_SIZE: 批次大小（3个样本）
  #   - INPUT_LENGTH: 序列长度（5个token）
  #   - 1, 1: 空间维度（T2T框架的约定格式，NLP任务中通常是1x1）
  inputs = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))

  # 生成随机目标数据（整数token ID）
  # 形状: [BATCH_SIZE, TARGET_LENGTH, 1, 1]
  targets = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))

  # 构建特征字典（T2T框架的数据格式）
  # tf.constant() 将numpy数组转换为TensorFlow常量张量
  features = {
      # targets: 目标序列（解码器需要生成的序列）
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      # target_space_id: 标量，表示目标空间ID（如翻译任务中表示目标语言）
      # 这里设为1，表示一种特定的目标空间/任务类型
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }

  # 如果有输入序列，将其添加到特征字典中
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")

  # 实例化并返回模型
  # model_cls(hparams, mode, p_hparams): 构造函数接受超参数、运行模式和问题超参数
  return model_cls(hparams, mode, p_hparams), features


def small_librispeech_model(param_overrides=None):
  """构建一个用于LibriSpeech（语音识别）任务的小型Transformer模型。

  LibriSpeech是一个英语语音数据集，特点是：
  - 输入是音频特征（梅尔频谱等），通常是浮点数矩阵
  - 输出是文本（字符或词片段）

  与文本翻译相比，语音输入的维度不同：
  - 文本输入: [batch, length, 1, 1]（token ID，整数）
  - 语音输入: [batch, length, 80, 3]（80维梅尔滤波器组特征，3个通道）
    80是梅尔频谱的频率维度，3是叠加帧数（通常是3帧堆叠作为特征）

  Args:
    param_overrides: 可选的参数覆盖字典，用于在基础配置上修改特定超参数。
                     例如 {"num_heads": 2} 将注意力头数改为2。
                     使用 hparams.set_hparam() 修改已有参数，
                     使用 hparams.add_hparam() 添加新参数。

  Returns:
    元组 (model, features)：
      model: 用于LibriSpeech任务的小型Transformer模型
      features: 特征字典，包含浮点型语音特征而非整数token ID
  """
  # 使用 transformer_small 超参数配置（比 tiny 稍大，更接近实用规模）
  hparams = transformer.transformer_small()

  # 缩小模型规模以加速测试
  hparams.hidden_size = 8      # 隐藏层维度
  hparams.filter_size = 32     # 前馈网络宽度
  hparams.num_heads = 1        # 注意力头数
  hparams.layer_prepostprocess_dropout = 0.0  # 关闭dropout

  # 获取LibriSpeech特有的问题超参数
  # Librispeech().get_hparams(hparams) 返回针对该任务的配置，
  # 包括音频特征大小、输入/输出模态等
  p_hparams = librispeech.Librispeech().get_hparams(hparams)

  # 将词表大小设为测试用的小值（实际LibriSpeech通常有数千个子词）
  p_hparams.vocab_size["targets"] = VOCAB_SIZE

  # 将问题超参数存入主超参数
  hparams.problem_hparams = p_hparams

  # 实例化Transformer模型
  model = transformer.Transformer(hparams, problem_hparams=p_hparams)

  # 处理可选的超参数覆盖
  if param_overrides is not None:
    # 确保 param_overrides 是字典格式
    assert isinstance(param_overrides, dict)
    for param_name in param_overrides:
      # 如果参数已存在，使用 set_hparam 修改
      if hasattr(hparams, param_name):
        hparams.set_hparam(param_name, param_overrides[param_name])
      else:
        # 如果参数不存在，使用 add_hparam 添加新参数
        hparams.add_hparam(param_name, param_overrides[param_name])

  # 生成随机语音特征输入（浮点数，非token ID）
  # np.random.rand() 生成 [0, 1) 范围内的均匀分布随机浮点数
  # 形状: [BATCH_SIZE, INPUT_LENGTH, 80, 3]
  #   - 80: 梅尔滤波器组的频率维度（Mel-filterbank features）
  #   - 3: 叠加帧（通常将连续3帧的特征堆叠）
  # .astype("float32"): 转换为32位浮点数（TensorFlow默认使用float32）
  inputs = np.random.rand(
      BATCH_SIZE, INPUT_LENGTH, 80, 3).astype("float32")  # modify for speech

  # 生成随机目标序列（字符/词片段的ID，整数）
  targets = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))

  # 构建特征字典
  # 注意：语音输入使用 tf.float32（与文本的 tf.int32 不同）
  features = {
      "inputs": tf.constant(inputs, dtype=tf.float32, name="inputs"),  # 语音特征（浮点）
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),  # 目标文本（整数ID）
      "target_space_id": tf.constant(1, dtype=tf.int32)  # 目标空间ID
  }
  return model, features


class TransformerTest(tf.test.TestCase):
  """Transformer模型的主要测试类。

  继承自 tf.test.TestCase，提供了TensorFlow特有的断言方法如：
  - self.assertEqual(): 检查两个值是否相等
  - self.assertAllClose(): 检查两个张量是否数值上近似相等（允许浮点误差）
  - self.test_session(): 创建TF 1.x风格的会话来执行图计算

  测试方法命名约定：以 "test" 开头的方法会被测试框架自动发现和执行。
  """

  def testTransformer(self, get_model_fn=None, p=None):
    """测试Transformer模型的基础前向传播。

    验证在训练模式下：
    1. 模型能够正常运行（不报错）
    2. 输出logits的形状符合预期：[BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE]
       - BATCH_SIZE: 批次大小
       - TARGET_LENGTH: 目标序列长度
       - 1, 1: 空间维度（T2T约定格式）
       - VOCAB_SIZE: 词表大小（最后一维是每个位置的词表分布）

    logits是未经softmax归一化的概率分布，通过argmax(logits, axis=-1)可得预测词ID。

    Args:
      get_model_fn: 可选的模型创建函数。如果为None，使用默认的 get_model()。
                    允许复用此测试逻辑来测试不同的模型变体（如语音模型）。
      p: 传递给 get_model_fn 的参数字典（超参数覆盖）。
    """
    # 使用提供的或默认的函数创建模型和特征
    if get_model_fn:
      model, features = get_model_fn(param_overrides=p)
    else:
      # 使用 transformer_small 超参数，创建标准Transformer
      model, features = get_model(transformer.transformer_small())

    # 调用模型：model(features) 执行前向传播
    # 返回 (logits, extra_loss)：
    #   logits: 预测分布，形状 [batch, length, 1, 1, vocab_size]
    #   _: 额外损失（这里不需要，用 _ 忽略）
    logits, _ = model(features)

    # 创建TF 1.x会话来执行计算图
    with self.test_session() as session:
      # 初始化所有变量（权重、偏置等）
      session.run(tf.global_variables_initializer())
      # 运行前向传播，获取实际的numpy数组结果
      res = session.run(logits)

    # 验证输出形状是否符合预期
    # (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE) = (3, 7, 1, 1, 10)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerLibrispeech(self, params=None):
    """测试语音识别（LibriSpeech）任务的Transformer。

    通过调用 testTransformer 并传入 small_librispeech_model 来复用测试逻辑。
    主要验证语音输入（浮点数矩阵）能够被正确处理。

    Args:
      params: 可选的超参数覆盖字典。
    """
    # 复用 testTransformer 的测试逻辑，但使用语音模型创建函数
    self.testTransformer(get_model_fn=small_librispeech_model, p=params)

  def testLibrispeechSlowVsFast(self, params=None):
    """测试LibriSpeech模型的慢速推理与快速推理结果一致性。

    验证：对于语音输入，带缓存的快速推理与不带缓存的慢速推理
    应该产生完全相同的输出序列。

    Args:
      params: 可选的超参数覆盖字典。
    """
    # 复用 testSlowVsFast 的测试逻辑，但使用语音模型
    self.testSlowVsFast(get_model_fn=small_librispeech_model, p=params)

  def testLibrispeechMultihead(self, params=None):
    """测试LibriSpeech模型使用多头注意力（2个头）时的正确性。

    多头注意力（Multi-Head Attention）将注意力分裂为多个"头"，
    每个头关注输入的不同子空间，最后拼接结果。
    这里测试将默认的1个头增加到2个头时模型是否仍然正常工作。
    """
    # 传入 num_heads=2 覆盖默认的单头配置
    self.testTransformerLibrispeech({"num_heads": 2})

  def testLibrispeechWithAreaAttention(self):
    """测试LibriSpeech模型使用区域注意力（Area Attention）时的正确性。

    区域注意力（Area Attention）是一种注意力变体，允许模型关注连续区域
    而不仅仅是单个位置。这对语音处理特别有用，因为音素往往跨越多帧。

    参数说明：
    - max_area_width: 最大区域宽度（2表示可以关注最多2个连续位置）
    - num_area_layers: 区域注意力层数
    - area_key_mode: 区域键向量的聚合方式（"mean"=取均值）
    - area_value_mode: 区域值向量的聚合方式（"sum"=求和）
    """
    self.testTransformerLibrispeech({"max_area_width": 2,
                                     "num_area_layers": 1,
                                     "area_key_mode": "mean",
                                     "area_value_mode": "sum"})

  def testTransformerRelative(self):
    """测试带相对位置编码的Transformer（Transformer-XL风格）。

    标准Transformer使用绝对位置编码（每个位置有固定的位置向量）。
    相对位置编码（Relative Position Encoding）不使用绝对位置，
    而是编码token对之间的相对距离，有以下优点：
    - 更好的长序列泛化能力
    - 理论上可以处理训练时未见过的序列长度

    使用 transformer_relative_tiny() 超参数来启用相对位置编码。
    """
    # 创建使用相对位置编码的极小Transformer
    model, features = get_model(transformer.transformer_relative_tiny())
    # 前向传播
    logits, _ = model(features)

    # 在会话中运行
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)

    # 验证输出形状不变（相对位置编码不改变输出形状）
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testSlowVsFast(self, get_model_fn=None, p=None):
    """验证慢速推理与快速推理（KV缓存）的输出完全一致。

    这是Transformer推理优化的关键测试。原理如下：

    慢速推理（_slow_greedy_infer）：
    - 每生成一个新token，重新计算所有历史token的注意力
    - 计算复杂度：O(n²)（n是序列长度）
    - 简单正确，但效率低

    快速推理（_greedy_infer）：
    - 维护一个KV缓存（Key-Value Cache），存储所有已计算的K和V矩阵
    - 每次只计算新token与所有历史token的注意力
    - 计算复杂度：O(n)（每步生成）
    - 效率高，但正确性需要验证

    测试流程：
    1. 先训练模型100步，使其参数有意义（而不是随机权重）
    2. 切换到推理模式
    3. 分别使用慢速和快速方法生成序列
    4. 验证两者结果完全相同（assertAllClose）

    Args:
      get_model_fn: 可选的模型创建函数。
      p: 传递给 get_model_fn 的参数字典。
    """
    # 创建模型（训练模式）
    if get_model_fn:
      model, features = get_model_fn(param_overrides=p)
    else:
      model, features = get_model(transformer.transformer_small())

    # 要额外解码的步数（在已有输入之后再生成3个token）
    decode_length = 3

    # ---- 训练阶段：让模型参数有意义 ----
    # 前向传播，获取logits
    out_logits, _ = model(features)
    # 去除多余的维度（从 [batch, len, 1, 1, vocab] 变为 [batch, len, vocab]）
    out_logits = tf.squeeze(out_logits, axis=[2, 3])

    # 计算交叉熵损失（分类损失）
    # sparse_softmax_cross_entropy_with_logits：计算 -log(softmax(logits)[label])
    # 先将 logits 和 labels 展平为2D/1D
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),   # [batch*len, vocab]
        labels=tf.reshape(features["targets"], [-1]))       # [batch*len]
    # 对所有位置的损失取均值
    loss = tf.reduce_mean(loss)

    # 定义Adam优化器（学习率0.001）并最小化损失
    # Adam是自适应学习率优化器，结合了动量（Momentum）和RMSProp
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 在会话中运行训练
    with self.test_session():
      tf.global_variables_initializer().run()  # 初始化变量
      # 运行100步训练，让模型学到有意义的参数
      for _ in range(100):
        apply_grad.run()

    # ---- 推理阶段：对比两种推理方式的结果 ----
    # 切换到推理模式（PREDICT）
    # 推理模式下：关闭dropout，使用缓存等
    model.set_mode(tf_estimator.ModeKeys.PREDICT)

    # tf.variable_scope(..., reuse=True)：允许在同一作用域内复用已有变量
    # 这是TF 1.x中共享权重的方式（两次推理使用相同的模型权重）
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      # 慢速贪心推理（无缓存，每步重新计算）
      # _slow_greedy_infer: 循环调用模型 decode_length 次
      greedy_result = model._slow_greedy_infer(
          features, decode_length)["outputs"]
      # 去除多余维度 [batch, input_len+decode_len, 1, 1] -> [batch, input_len+decode_len]
      greedy_result = tf.squeeze(greedy_result, axis=[2, 3])

      # 快速贪心推理（有缓存，每步只计算新token）
      # _greedy_infer: 使用KV缓存的高效推理
      fast_result = model._greedy_infer(features, decode_length)["outputs"]
      # 注：fast_result 已经是 [batch, input_len+decode_len] 格式，不需要squeeze

    # 在会话中求值（eval() 执行计算图并返回numpy数组）
    with self.test_session():
      greedy_res = greedy_result.eval()
      fast_res = fast_result.eval()

    # 验证形状：输出长度 = 输入长度 + 解码长度
    self.assertEqual(fast_res.shape, (BATCH_SIZE, INPUT_LENGTH + decode_length))
    # 验证数值相等（两种方式应产生完全相同的结果）
    # assertAllClose 允许微小的浮点误差（实际上对离散序列结果应该完全相同）
    self.assertAllClose(greedy_res, fast_res)

  def testSlowVsFastNoInput(self):
    """验证无输入（语言模型）情况下慢速与快速推理结果一致。

    语言模型（Language Model）与seq2seq模型的区别：
    - seq2seq（has_input=True）：有编码器输入，解码器做条件生成
    - 语言模型（has_input=False）：没有编码器，只有解码器自回归生成

    无输入时解码行为不同：
    - 有输入时：输出长度 = INPUT_LENGTH + decode_length（从输入末尾开始生成）
    - 无输入时：输出长度 = decode_length（从头开始生成）

    本测试验证语言模型情况下两种推理方式的结果一致性。
    """
    # 创建无输入序列的模型（语言模型配置）
    model, features = get_model(
        transformer.transformer_small(), has_input=False)

    decode_length = 3  # 要生成的token数量

    # 训练模型（与 testSlowVsFast 相同的训练流程）
    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    # 切换到推理模式
    model.set_mode(tf_estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      # 慢速推理（使用 slow_result 而不是 greedy_result 命名）
      slow_result = model._slow_greedy_infer(
          features, decode_length)["outputs"]
      slow_result = tf.squeeze(slow_result, axis=[2, 3])

      # 快速推理
      fast_result = model._greedy_infer(features, decode_length)["outputs"]

    with self.test_session():
      slow_res = slow_result.eval()
      fast_res = fast_result.eval()

    # 无输入时，输出长度 = decode_length（而不是 INPUT_LENGTH + decode_length）
    self.assertEqual(slow_res.shape, (BATCH_SIZE, decode_length))
    # 验证两种方式结果一致
    self.assertAllClose(slow_res, fast_res)

  def testBeamDecodeWithRelativeAttention(self):
    """测试使用束搜索解码结合相对位置编码的Transformer。

    束搜索（Beam Search）解码策略：
    - 每一步不只保留最优的一个候选（贪心搜索），而是保留top-k个候选序列
    - beam_size=4 表示同时追踪4条候选路径
    - top_beams=1 表示最终只返回最好的那条路径
    - alpha=1.0 是长度惩罚系数（防止模型倾向于生成过短的序列）

    与贪心搜索相比，束搜索通常能生成质量更高的序列，
    但计算成本是贪心搜索的 beam_size 倍。

    注意：此测试只验证运行不报错，不验证具体形状，
    原因是解码可能在 decode_length 之前遇到终止符（EOS）。
    """
    decode_length = 2

    # 使用相对位置编码的配置创建模型
    model, features = get_model(transformer.transformer_relative_tiny())
    # 切换到推理模式
    model.set_mode(tf_estimator.ModeKeys.PREDICT)

    # 执行束搜索解码
    # _beam_decode: 使用KV缓存的高效束搜索
    beam_result = model._beam_decode(
        features, decode_length, beam_size=4, top_beams=1,
        alpha=1.0)["outputs"]

    # 只验证运行不报错（eval()能成功执行）
    with self.test_session():
      tf.global_variables_initializer().run()
      beam_result.eval()

    # TODO(petershaw): 这个测试是不稳定的（flaky），因为解码可能在
    # 到达预期长度之前就遇到EOS（终止符）而停止。
    # self.assertEqual(beam_res.shape,
    #                  (BATCH_SIZE, INPUT_LENGTH + decode_length))

  def testBeamVsFast(self):
    """验证慢速束搜索与快速束搜索（KV缓存）结果一致。

    类似 testSlowVsFast，但针对束搜索解码：
    - _beam_decode_slow：每步重新计算所有历史token的注意力（慢速）
    - _beam_decode：使用KV缓存的高效束搜索（快速）

    两者应该产生完全相同的输出序列。

    测试流程：
    1. 训练模型100步
    2. 分别用慢速和快速束搜索生成序列
    3. 验证两者结果相同
    """
    model, features = get_model(transformer.transformer_small())

    decode_length = 2

    # 训练模型（使权重有意义）
    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    # 切换到推理模式
    model.set_mode(tf_estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      # 慢速束搜索（每步重新计算全部注意力）
      beam_result = model._beam_decode_slow(
          features,
          decode_length,
          beam_size=4,    # 保留4条候选路径
          top_beams=1,    # 只返回最好的1条
          alpha=1.0)["outputs"]  # 长度惩罚系数

      # 快速束搜索（使用KV缓存）
      fast_result = model._beam_decode(
          features,
          decode_length,
          beam_size=4,
          top_beams=1,
          alpha=1.0)["outputs"]

    with self.test_session():
      beam_res = beam_result.eval()
      fast_res = fast_result.eval()

    # 验证两种束搜索方式结果相同
    self.assertAllClose(beam_res, fast_res)

  def testTransformerWithoutProblem(self):
    """测试不使用Problem配置时的Transformer（直接使用嵌入向量作为输入）。

    正常使用场景：
    - 输入是token ID（整数），模型内部通过嵌入层转化为向量
    - 需要 problem_hparams 配置来设定词表大小等

    此测试场景（无Problem）：
    - 输入直接是嵌入向量（浮点数），跳过嵌入层
    - 不需要 problem_hparams
    - 适用于需要自定义输入预处理的场景

    输入格式：
    - 文本token输入: [batch, length, 1, 1] （整数）
    - 嵌入向量输入: [batch, length, 1, hidden_size] （浮点数）
    输出格式：
    - body输出（无problem时）: [batch, length, 1, hidden_size] （浮点数，未经分类头）
    """
    # transformer_test() 是专为此类测试设计的超参数配置
    hparams = transformer.transformer_test()

    # 生成随机的嵌入向量输入（浮点数，已经是嵌入表示）
    # np.random.random_sample() 生成 [0, 1) 范围内的均匀随机浮点数
    # 形状: [batch, length, 1, hidden_size]
    embedded_inputs = np.random.random_sample(
        (BATCH_SIZE, INPUT_LENGTH, 1, hparams.hidden_size))
    embedded_targets = np.random.random_sample(
        (BATCH_SIZE, TARGET_LENGTH, 1, hparams.hidden_size))

    # 构建特征字典（直接使用浮点嵌入，不是整数token ID）
    transformed_features = {
        "inputs": tf.constant(embedded_inputs, dtype=tf.float32),
        "targets": tf.constant(embedded_targets, dtype=tf.float32)
    }

    # 实例化Transformer（不传 problem_hparams，模型直接处理嵌入向量）
    model = transformer.Transformer(hparams)
    # 前向传播
    body_out, _ = model(transformed_features)

    # 验证输出形状：无problem时，输出是隐藏状态而不是词表分布
    # 形状: [BATCH_SIZE, TARGET_LENGTH, 1, hidden_size]
    self.assertAllEqual(
        body_out.get_shape().as_list(),
        [BATCH_SIZE, TARGET_LENGTH, 1, hparams.hidden_size])

  def testTransformerWithEncoderDecoderAttentionLoss(self):
    """测试带监督注意力损失（Supervised Attention Loss）的Transformer。

    监督注意力（Supervised/Guided Attention）是一种训练技术：
    - 除了常规的序列生成损失外，还额外监督注意力权重
    - 通过提供"期望的注意力矩阵"（expected_attentions），
      引导编码器-解码器注意力学习特定的对齐模式
    - 例如：在语音合成中，强制注意力权重沿对角线分布

    测试目的：验证额外的注意力损失能够被正确计算（形状为标量）。

    使用 transformer_supervised_attention() 超参数，该配置启用了
    supervised attention loss功能。
    """
    # 使用支持监督注意力的超参数创建模型
    model, features = get_model(
        transformer.transformer_supervised_attention())

    # 生成随机的"期望注意力权重"矩阵
    # 形状: [BATCH_SIZE, TARGET_LENGTH, INPUT_LENGTH]
    # 含义: 期望解码器第i步时，对编码器第j个位置的注意力权重
    expected_attention_weights = np.random.random_sample(
        size=(BATCH_SIZE, TARGET_LENGTH, INPUT_LENGTH))

    # 将期望注意力添加到特征字典
    features["expected_attentions"] = tf.constant(
        expected_attention_weights, dtype=tf.float32)

    # 前向传播，extra_loss 中包含额外的注意力损失
    _, extra_loss = model(features)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      # 从额外损失字典中获取 "attention_loss" 的值
      res = session.run(extra_loss["attention_loss"])

    # 验证注意力损失是标量（形状为空元组 ()）
    # 因为损失是所有样本、所有位置的平均值，所以是标量
    self.assertEqual(res.shape, ())

  def _create_greedy_infer_model(self):
    """创建一个用于贪心推理测试的已训练模型。

    这是一个辅助方法（以 _ 开头表示"私有"），被多个测试方法复用。
    避免在多个测试中重复相同的模型创建和训练代码。

    流程：
    1. 创建小型Transformer模型（训练模式）
    2. 训练100步（使参数有实际意义）
    3. 切换到推理模式（PREDICT）
    4. 返回模型和特征

    Returns:
      model: 经过短暂训练后处于推理模式的Transformer模型。
      features: 包含随机输入数据的特征字典。
    """
    # 创建模型（训练模式）
    model, features = get_model(transformer.transformer_small())

    # 训练模型（与 testSlowVsFast 中相同的训练流程）
    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    # 切换到推理模式
    model.set_mode(tf_estimator.ModeKeys.PREDICT)

    return model, features

  def testGreedySlowTPUVsNonTPU(self):
    """验证慢速贪心推理在TPU和非TPU上结果相同。

    T2T支持在TPU（Tensor Processing Unit）上运行，
    TPU版本可能使用不同的计算图结构（如 while_loop 代替 py_func）。
    本测试验证 TPU 和非 TPU 版本的慢速推理产生相同的结果。

    慢速推理的TPU版本（_slow_greedy_infer_tpu）：
    - 使用 tf.while_loop 实现自回归解码（TPU友好）
    - 非TPU版本（_slow_greedy_infer）可能使用Python循环

    测试验证：
    - 两者的输出形状相同
    - 两者的输出数值相同（assertAllClose）
    """
    decode_length = 3

    # 使用辅助方法创建已训练的推理模型
    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      # 非TPU版本的慢速贪心推理
      slow_result_non_tpu = model._slow_greedy_infer(
          features, decode_length)["outputs"]
      slow_result_non_tpu = tf.squeeze(slow_result_non_tpu, axis=[2, 3])

      # TPU版本的慢速贪心推理
      slow_result_tpu = model._slow_greedy_infer_tpu(
          features, decode_length)["outputs"]
      slow_result_tpu = tf.squeeze(slow_result_tpu, axis=[2, 3])

    with self.test_session():
      slow_non_tpu_res = slow_result_non_tpu.eval()
      slow_tpu_res = slow_result_tpu.eval()

    # 验证TPU版本的输出形状
    self.assertEqual(slow_tpu_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    # 验证TPU和非TPU版本结果数值相同
    self.assertAllClose(slow_tpu_res, slow_non_tpu_res)

  def testGreedyFastTPUVsNonTPU(self):
    """验证快速贪心推理（KV缓存）在TPU和非TPU上结果相同。

    与 testGreedySlowTPUVsNonTPU 类似，但针对快速推理（带KV缓存）。
    _greedy_infer 通过 use_tpu 参数控制使用哪种实现：
    - use_tpu=False：CPU/GPU实现（使用TensorArray等）
    - use_tpu=True：TPU实现（使用固定形状张量，避免动态形状）

    TPU的特殊要求：
    - 不支持动态形状（所有张量形状必须在编译时确定）
    - 不支持某些操作（如 tf.py_func）
    - 通常使用 tf.while_loop 而不是Python循环
    """
    decode_length = 3

    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      # 非TPU快速推理（use_tpu=False，默认）
      fast_result_non_tpu = model._greedy_infer(
          features, decode_length, use_tpu=False)["outputs"]

      # TPU快速推理（use_tpu=True，使用TPU友好的实现）
      fast_result_tpu = model._greedy_infer(
          features, decode_length, use_tpu=True)["outputs"]

    with self.test_session():
      fast_non_tpu_res = fast_result_non_tpu.eval()
      fast_tpu_res = fast_result_tpu.eval()

    # 验证TPU版本输出形状
    self.assertEqual(fast_tpu_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    # 验证TPU和非TPU版本结果数值相同
    self.assertAllClose(fast_tpu_res, fast_non_tpu_res)

  def testGreedyTPUSlowVsFast(self):
    """验证TPU上慢速推理与快速推理（KV缓存）结果一致。

    结合前两个TPU测试的验证：
    - testGreedySlowTPUVsNonTPU：验证慢速 TPU == 慢速 非TPU
    - testGreedyFastTPUVsNonTPU：验证快速 TPU == 快速 非TPU
    - 本测试：验证 慢速 TPU == 快速 TPU

    从逻辑上说，如果前两个测试通过，本测试也应通过。
    但显式测试TPU上慢速vs快速的一致性是额外的安全保证。
    """
    decode_length = 3

    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      # TPU版本的慢速推理
      slow_result = model._slow_greedy_infer_tpu(
          features, decode_length)["outputs"]
      slow_result = tf.squeeze(slow_result, axis=[2, 3])

      # TPU版本的快速推理（use_tpu=True）
      fast_result = model._greedy_infer(
          features, decode_length, use_tpu=True)["outputs"]

    with self.test_session():
      slow_res = slow_result.eval()
      fast_res = fast_result.eval()

    # 验证TPU快速推理的输出形状
    self.assertEqual(fast_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    # 验证TPU上慢速与快速推理结果相同
    self.assertAllClose(fast_res, slow_res)


class TransformerScorerTest(tf.test.TestCase):
  """TransformerScorer的测试类。

  TransformerScorer是Transformer的一个变体，主要用于：
  - 不生成新序列，而是对给定的输入-输出序列对计算概率得分
  - 常用于重排序（Reranking）：先生成多个候选序列，再用Scorer对每个候选打分
  - 也用于序列分类任务的概率计算

  与标准Transformer的区别：
  - 标准Transformer.infer()：自回归生成新序列
  - TransformerScorer.infer()：计算已知序列的对数概率（log probability）
  """

  def testReturnsScores(self):
    """测试TransformerScorer.infer()能够返回正确格式的得分和输出。

    验证：
    1. infer() 的返回字典中包含 "outputs" 键（token ID序列）
    2. infer() 的返回字典中包含 "scores" 键（序列对数概率）
    3. scores 的形状是 [BATCH_SIZE]（每个样本一个标量分数）
    4. outputs 的形状是 [BATCH_SIZE, TARGET_LENGTH]
    """
    # 在推理模式下创建TransformerScorer
    # 注意：使用 model_cls=transformer.TransformerScorer 指定模型类
    model, features = get_model(
        mode=tf_estimator.ModeKeys.PREDICT,
        model_cls=transformer.TransformerScorer)

    # 运行推理
    infer_out = model.infer(features)

    # 验证返回字典中包含必要的键
    self.assertTrue("outputs" in infer_out)   # 必须有输出序列
    self.assertTrue("scores" in infer_out)    # 必须有得分

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      # 执行计算图，得到实际数组
      infer_out = session.run(infer_out)

      # 验证 scores 形状：每个样本一个标量分数
      self.assertEqual((BATCH_SIZE,), infer_out["scores"].shape)
      # 验证 outputs 形状：每个样本的目标序列（不含额外生成）
      self.assertEqual((BATCH_SIZE, TARGET_LENGTH), infer_out["outputs"].shape)

  def testVarNames(self):
    """测试TransformerScorer的变量名与标准Transformer完全相同。

    这是为了确保 TransformerScorer 和 Transformer 可以共享权重：
    - 如果两者的变量名相同，可以将Transformer训练好的权重直接加载到Scorer中
    - 这使得"训练Transformer，然后用Scorer打分"的工作流成为可能

    测试方法：
    1. 在独立的TF图（tf.Graph）中创建每个模型
    2. 收集各模型的所有变量名
    3. 验证三组变量名两两相同：
       - TransformerScorer（推理模式）的变量名
       - TransformerScorer（评估模式）的变量名
       - 标准Transformer（评估模式）的变量名

    使用 tf.Graph().as_default() 创建独立的计算图，
    避免不同模型的变量互相污染。
    """
    # 创建第一个图：推理模式的TransformerScorer
    with tf.Graph().as_default():
      model, features = get_model(
          mode=tf_estimator.ModeKeys.PREDICT,
          model_cls=transformer.TransformerScorer)
      _ = model.infer(features)
      # 收集此图中所有全局变量的名称
      scorer_vars = [v.name for v in tf.global_variables()]

    # 创建第二个图：评估模式的TransformerScorer
    with tf.Graph().as_default():
      model, features = get_model(
          mode=tf_estimator.ModeKeys.EVAL,
          model_cls=transformer.TransformerScorer)
      _ = model(features)
      scorer_eval_vars = [v.name for v in tf.global_variables()]

    # 创建第三个图：评估模式的标准Transformer
    with tf.Graph().as_default():
      model, features = get_model(
          mode=tf_estimator.ModeKeys.EVAL,
          model_cls=transformer.Transformer)
      _ = model(features)
      transformer_vars = [v.name for v in tf.global_variables()]

    # 验证三组变量名排序后完全相同
    # sorted() 先排序再比较，避免顺序差异导致误判
    self.assertEqual(sorted(scorer_vars), sorted(transformer_vars))
    self.assertEqual(sorted(scorer_eval_vars), sorted(transformer_vars))


# Python脚本入口点
# 当直接运行此文件（python transformer_test.py）时执行测试
# 而不是被 import 时执行（import时 __name__ != "__main__"）
if __name__ == "__main__":
  # tf.test.main() 会自动发现并运行所有以 "test" 开头的方法
  tf.test.main()
