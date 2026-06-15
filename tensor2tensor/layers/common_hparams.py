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

"""多模型通用的超参数 (Hyperparameter) 定义与取值范围。
   
   超参数（HParams）是在训练开始前人为设置的参数，用于控制模型的结构和训练过程。
   与模型训练过程中自动学习的"模型参数"（如权重）不同，超参数是固定不变的。
   例如：学习率、批次大小、层数等。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# 导入 Python 2/3 兼容的 zip 函数
from six.moves import zip  # pylint: disable=redefined-builtin
# 导入超参数管理工具
from tensor2tensor.utils import hparam
# 导入模型注册器，用于将超参数配置注册到全局配置系统中
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_hparams("basic_1")  # 将此函数注册为名称 "basic_1" 的超参数配置
def basic_params1():
  """定义一套基础超参数集合，适用于大多数模型的默认配置。
  
  这些超参数涵盖了训练流程的方方面面：
  - 批次大小和数据读取
  - 模型结构（层数、隐藏层大小）
  - 优化器设置（Adam、Adafactor等）
  - 学习率调度（预热、衰减）
  - 正则化（Dropout、权重衰减）
  - 序列长度控制
  - 视频处理参数
  等等
  """
  return hparam.HParams(
      # ========== 批次大小配置 ==========
      # 如果问题使用变长序列（见 problem.batch_size_means_tokens()），
      # 则此参数表示每个 GPU 或 TPU 核心每批次的 token 数量。
      # 否则，表示每个 GPU 或 TPU 核心的样本数量。
      batch_size=4096,
      # 数据 shuffle 缓冲区大小，训练时随机打乱数据的队列长度
      batch_shuffle_size=512,
      # 如果为 True，即使是变长序列，也直接用 batch_size 作为实际批次大小
      # （而不是将其视为每批 token 数）。
      use_fixed_batch_size=False,

      # ========== 模型结构参数 ==========
      # 隐藏层（网络层）的数量
      num_hidden_layers=4,
      # 卷积核的高度（用于卷积模型）
      kernel_height=3,
      # 卷积核的宽度（用于卷积模型）
      kernel_width=1,
      # 隐藏层的维度大小（也叫模型维度 d_model）
      # 这是 Transformer 中 embedding 向量和每层输出的维度
      hidden_size=64,
      # 压缩步骤数量（用于需要序列压缩的模型）
      compress_steps=0,

      # ========== 正则化参数 ==========
      # Dropout 概率。所有以 "dropout" 结尾的超参数在非训练模式下会自动设为 0.0。
      # Dropout 是一种正则化技术，训练时随机将部分神经元输出置零，
      # 可以防止过拟合。
      dropout=0.2,
      # 梯度裁剪的范数上界，防止梯度爆炸
      # 当梯度的 L2 范数超过此值时，对梯度进行缩放
      clip_grad_norm=2.0,
      # 梯度噪声的标准差。添加高斯噪声到梯度，可以帮助走出局部最优
      grad_noise_scale=0.0,
      # 是否在 TensorBoard 中记录梯度摘要信息（用于调试）
      summarize_grads=False,
      # 是否开启 MLPerf 基准测试模式
      mlperf_mode=False,
      # 是否记录每个变量的名称和大小（用于调试模型结构）
      summarize_vars=False,

      # ========== 参数初始化 ==========
      # 权重初始化方式，可选 "orthogonal"（正交初始化）、"uniform"（均匀分布）等
      # 正交初始化有助于保持梯度的稳定传播
      initializer="orthogonal",
      # 初始化时的增益系数，乘以初始化的标准差
      initializer_gain=1.5,

      # ========== 损失函数 ==========
      # 标签平滑系数。标签平滑是一种正则化方法：
      # 将 one-hot 标签 [0,0,1,0] 转换为 [epsilon/K, ..., 1-epsilon, ..., epsilon/K]
      # 这样可以防止模型过于自信，提高泛化能力。
      # 0.1 表示将目标标签从 1.0 平滑到 0.9，将其他标签从 0.0 平滑到 0.1/K
      label_smoothing=0.1,

      # ========== 优化器配置 ==========
      # 使用的优化器类型，如 "adam"、"adagrad"、"sgd" 等
      optimizer="adam",
      # Adam 优化器的 epsilon 参数，防止除以零，增加数值稳定性
      optimizer_adam_epsilon=1e-6,
      # Adam 优化器的 beta1 参数，用于计算梯度的一阶矩估计（历史梯度的指数加权平均）
      optimizer_adam_beta1=0.85,
      # Adam 优化器的 beta2 参数，用于计算梯度的二阶矩估计（历史梯度平方的指数加权平均）
      optimizer_adam_beta2=0.997,
      # Momentum 优化器的动量系数
      optimizer_momentum_momentum=0.9,
      # 是否使用 Nesterov 动量
      optimizer_momentum_nesterov=False,
      # Adafactor 优化器参数（beta1=0 表示不使用一阶矩估计）
      optimizer_adafactor_beta1=0.0,
      optimizer_adafactor_beta2=0.999,
      # Adafactor 是否使用分解的参数矩阵来节省内存
      optimizer_adafactor_factored=True,
      # Adafactor 二阶矩的衰减类型，"pow" 表示幂次衰减
      optimizer_adafactor_decay_type="pow",
      optimizer_adafactor_memory_exponent=0.8,
      # Adafactor 梯度裁剪阈值
      optimizer_adafactor_clipping_threshold=1.0,
      # Adafactor 是否按参数规模进行乘法缩放（类似于自适应学习率）
      optimizer_adafactor_multiply_by_parameter_scale=True,
      # 多步优化器（梯度累积）的步数。0 表示不使用梯度累积。
      # 梯度累积可以在 GPU 显存不足时模拟更大的批次大小
      optimizer_multistep_accumulate_steps=0,
      
      # ========== 混合精度训练 ==========
      # 损失缩放策略，主要用于混合精度训练（float16+float32）。
      # 混合精度使用 float16 进行前向传播，float32 进行梯度更新，
      # 但 float16 容易下溢，需要对损失进行缩放。
      # 目前混合精度训练只支持 "exponential"（指数）缩放
      mixed_precision_optimizer_loss_scaler="exponential",
      # 初始损失缩放值（2^15 = 32768）
      mixed_precision_optimizer_init_loss_scale=2**15,
      # 是否将未计算的梯度清零，以便创建相应的优化器槽（slots）
      # 在不同模型头部之间共享检查点时很有用
      optimizer_zero_grads=False,
      
      # ========== 正则化（权重） ==========
      # L2 权重衰减系数，防止权重过大（过拟合）
      weight_decay=1e-6,
      # 权重噪声的标准差，训练时向权重添加高斯噪声（一种正则化方法）
      weight_noise=0.0,

      # ========== 学习率调度 ==========
      # 学习率调度方式，定义为多个命名函数的乘积。
      # 可用函数见 learning_rate._LEARNING_RATE_FUNCTIONS
      # 示例："constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size"
      # "legacy" 表示使用旧版调度方式（由 learning_rate_decay_scheme 控制）
      learning_rate_schedule="legacy",
      # 学习率常数因子
      learning_rate_constant=1.0,
      # 如果 learning_rate_schedule=="legacy"，则由此参数指定衰减方案：
      # "none" - 无衰减
      # "sqrt" - 平方根衰减
      # "noam" - Transformer 论文中使用的 Noam 调度（含预热）
      # "exp"  - 指数衰减
      # "cosine" - 余弦衰减
      # 预热阶段始终使用指数增长（"noam" 除外）
      learning_rate_decay_scheme="none",
      # 指数衰减的步数间隔（learning_rate_decay_scheme=="exp" 时使用）
      learning_rate_decay_steps=5000,
      # 是否使用阶梯式衰减（每隔 decay_steps 步才进行一次衰减）
      learning_rate_decay_staircase=False,
      # 学习率的最小值下界，防止学习率衰减到太小
      learning_rate_minimum=None,
      # 指数衰减的衰减率
      learning_rate_decay_rate=1.0,
      # 学习率预热（warmup）步数。
      # 预热阶段学习率从 0 线性或指数增长到设定值，
      # 有助于模型在训练初期的稳定性
      learning_rate_warmup_steps=100,
      # 余弦退火调度的周期步数
      learning_rate_cosine_cycle_steps=250000,
      # 基础学习率
      learning_rate=0.1,

      # ========== 推理采样策略 ==========
      # 采样方式："argmax"（贪心，取概率最大的词）或 "random"（随机采样）
      sampling_method="argmax",  # "argmax" or "random"
      # 随机采样时的温度参数，温度越高分布越均匀，越低越集中于高概率词
      sampling_temp=1.0,  # temperature for sampling
      # 如果 >0，只从概率最高的 k 个候选词中采样（Top-K 采样）
      sampling_keep_top_k=-1,  # If >0, ignore all but the top k logits
      # 分块计算 logits，节省内存（适用于词表很大时）
      factored_logits=False,
      # 嵌入向量乘以 sqrt(hidden_size) 的模式，"sqrt_depth" 是 Transformer 标准做法
      # 这样做是为了让词嵌入的幅度与位置编码相匹配
      multiply_embedding_mode="sqrt_depth",

      # ========== 混合专家模型 (MoE) 参数 ==========
      # MoE (Mixture of Experts) 是一种稀疏激活的模型结构，
      # 每个 token 只激活部分专家子网络，从而大幅扩展模型容量但不增加计算量
      # 每个专家的隐藏层大小（逗号分隔的字符串）
      moe_hidden_sizes="2048",  # hidden layer sizes (comma-separated)
      # 每层的专家数量
      moe_num_experts=64,  # number of experts per layer
      # 每个 token 使用几个专家（Top-K 路由）
      moe_k=2,  # how many experts to use for each batch element
      # MoE 负载均衡损失的系数（鼓励各专家均匀受到分配）
      moe_loss_coef=1e-2,

      # ========== 层预处理/后处理序列 ==========
      # 用于 common_layers.layer_preprocess 和 common_layers.layer_postprocess
      # 每个字符代表一个操作：
      #   "none" - 不做任何处理
      #   "d"    - 应用 Dropout
      #   "n"    - 应用归一化（LayerNorm/BatchNorm 等，由 norm_type 控制）
      #   "a"    - 加上层输入（残差连接 Residual Connection，只在后处理中有效）
      # "none" 字符串代替空字符串，因为空字符串在超参数调优中会有问题。
      # 当前设置 ("none", "dan") 是原始 Transformer 论文的版本（先计算再归一化）。
      # ("n", "da") 是 Pre-LayerNorm 变体，对难训练的模型效果更好。
      layer_preprocess_sequence="none",
      # 后处理顺序："d"=dropout, "a"=残差连接, "n"=层归一化
      # 即：先 dropout，再加残差，再做层归一化
      layer_postprocess_sequence="dan",
      # 层预处理/后处理中使用的 dropout 概率
      layer_prepostprocess_dropout=0.1,
      # Dropout 的广播维度（逗号分隔的整数列表）
      # 设为 "1" 可以节省内存（在 batch 维度广播 dropout）
      # 见 common_layers.dropout_with_broadcast_dims()
      layer_prepostprocess_dropout_broadcast_dims="",
      # 符号（token）级别的 dropout 概率（在 embedding 之前随机置零某些 token）
      symbol_dropout=0.0,

      # ========== 归一化 ==========
      # 归一化类型："batch"（批归一化）, "layer"（层归一化）, "noam", "none"
      # LayerNorm 是 Transformer 的标准选择，对每个位置的特征进行归一化
      norm_type="layer",  # "batch", layer", "noam", "none".
      # 归一化函数的 epsilon 参数，防止除以零，增加数值稳定性
      norm_epsilon=1e-6,
      # 词表大小将被对齐到此值的倍数（默认不对齐）
      vocab_divisor=1,

      # ========== 序列长度过滤 ==========
      # 训练时，丢弃输入和目标都短于 min_length 的序列
      min_length=0,
      # 训练时，丢弃输入或目标长于 max_length 的序列。
      # 如果 max_length==0，则使用 batch_size 作为最大长度。
      max_length=0,
      # 是否在读取时动态打包多个短样本到一个固定长度的序列
      # （Pack examples on the fly，提高 GPU/TPU 利用率）
      pack_dataset=False,
      # 是否使用标准 TensorFlow 之外的自定义 Op
      use_custom_ops=True,
      # 将目标序列在第一个轴上切分成若干固定长度的块（0 表示不切分）
      split_targets_chunk_length=0,
      split_targets_max_chunks=100,
      split_targets_strided_training=False,

      # ========== 长度分桶（Length Bucketing） ==========
      # 最小桶的最大序列长度。
      # 设置过高会浪费内存在短序列的填充上；
      # 设置过低会导致批次 shuffle 队列很长。
      min_length_bucket=8,
      # 控制长度桶的数量。桶的最大长度从 min_bucket_length 增长到
      # (max_length 或 batch_size)，每步大约乘以 length_bucket_step。
      # 1.1 表示每个桶的最大长度是上一个桶的 1.1 倍
      length_bucket_step=1.1,
      # 如果为 True，在评估时丢弃超过 max_length 的序列
      # 注意：这会影响评估指标的有效性
      eval_drop_long_sequences=False,
      # 如果为 True，在评估时使用自回归（逐步生成）而不是 Teacher Forcing 模式
      eval_run_autoregressive=False,

      # ========== 嵌入层共享 ==========
      # （针对 symbol modality）如果为 True，共享所有的输入 embedding、
      # 目标 embedding 和 softmax 权重矩阵。
      # 这是 Transformer 的标准做法，可以减少参数数量并提高性能。
      shared_embedding_and_softmax_weights=False,
      # （针对 symbol modality）如果为 True，共享输入 embedding 和目标 embedding。
      shared_embedding=False,
      # （针对 symbol modality）embedding 矩阵分片数量（用于超大词表）
      symbol_modality_num_shards=1,

      # ========== 特征变换 ==========
      # 特征变换是可选字典，包含特征名称（str）和变换函数（function）的键值对。
      # 如果不指定，T2TModel 会根据特征的 modality 自动应用默认变换。
      # bottom 适用于所有特征；loss, top, weights_fn 仅适用于目标特征。
      bottom={},  # 输入变换（从原始特征到 embedding）
      loss={},    # 损失函数变换
      name={},    # 变量作用域名称（历史遗留参数，未来可能移除）
      top={},     # 输出变换（从隐藏状态到 logits）
      weights_fn={},  # 损失权重函数

      # ========== 序列长度截断 ==========
      # 输入序列的最大长度，超过此长度的序列会被截断
      # 0 或负值表示不截断
      max_input_seq_length=0,
      # 目标序列的最大长度，超过此长度的序列会被截断
      max_target_seq_length=0,
      # 如果非零，在读取样本时将目标序列切分成此长度的片段
      # 用于语言模型问题中处理超长固定长度样本
      # 例如：样本长度为 65536，通过设置此参数切分为 64 个长度 1024 的样本
      split_to_length=0,

      # ========== 视频参数 ==========
      # 输入帧数（用于视频预测模型）
      video_num_input_frames=1,
      # 目标帧数（要预测的未来帧数）
      video_num_target_frames=1,

      # ========== 序列到序列的语言模型模式 ==========
      # 可选值：
      # "none" - 不将输入前置到目标序列（标准 seq2seq 模式）
      # "prepend_inputs_masked_attention"
      #     将目标替换为 tf.concat([inputs, [0], targets], axis=1)
      #     即：在输入后面接一个填充 token，再接目标序列。
      #     在整个拼接序列上使用掩码自注意力（因果掩码，只看左侧）。
      #     训练时在整个序列上计算损失，评估时只计算目标部分的指标。
      # "prepend_inputs_full_attention"
      #     类似上一选项，但输入部分中的每个位置可以看到整个输入部分
      #     （双向注意力），这样输入部分不需要自回归地预测，降低了难度。
      prepend_mode="none",

      # ========== 计划采样 (Scheduled Sampling) ==========
      # 计划采样是自回归模型的一种训练技巧：
      # 以一定概率 (scheduled_sampling_prob) 额外运行一步，
      # 使用生成的输出而非真实标签作为自回归目标，
      # 可以减少训练和推理时的分布偏差（exposure bias）。
      # 0.0 表示关闭此功能。
      scheduled_sampling_prob=0.0,
      # 采样模式："parallel"（并行采样）或 "sequential"（顺序采样）
      scheduled_sampling_method="parallel",  # parallel or sequential.
      # 计划采样概率的预热步数（前 N 步从 0 指数增长到设定概率）
      scheduled_sampling_warmup_steps=50000,
      # 每步中使用真实标签的比例（1.0 表示全用真实标签，0.0 表示全用生成结果）
      scheduled_sampling_gold_mixin_prob=0.5,
      # 每步运行的额外自回归步骤数
      scheduled_sampling_num_passes=1,
      # 预热调度类型："exp"（指数）、"linear"（线性）或 "sigmoid"（S 形）
      scheduled_sampling_warmup_schedule="exp",  # exp, linear, or sigmoid.

      # ========== 分布式训练 ==========
      # 是否将变量按菊花链（daisy chain）方式在设备间复制，
      # 而不是依赖 TensorFlow 的自动放置。
      # 对多设备训练的性能很重要。
      # 注意：使用动态循环的递归模型必须设置为 False。
      daisy_chain_variables=True,
      # 如果为 True，在预测（PREDICT）模式下不使用"只处理最后位置"的优化
      force_full_predict=False,
      # 如果为 True，设置为纯模型并行模式，只有一个数据分片
      no_data_parallelism=False,

      # ========== 数值精度 ==========
      # 激活值的数据类型："float32" 或 "bfloat16"
      # bfloat16 目前只支持 TPU，可以降低激活内存占用，对模型质量影响不大。
      # 可以在 TPU 上用 bfloat16 训练，在 CPU/GPU 上用 float32 评估。
      activation_dtype="float32",
      # 模型参数（权重）的数据类型："float32" 或 "bfloat16"
      # bfloat16 目前只支持 adafactor 优化器，可以训练更大的模型（节省内存）。
      # 权重编码为 (w*128)^8 的形式，使用伪随机舍入。
      weight_dtype="float32",

      # ========== 迁移学习 ==========
      # 预训练模型的检查点目录。
      # 只在新开始的运行中使用（已有检查点时不使用）。
      # 预训练模型中没有的参数会随机初始化，多余的参数会被忽略。
      pretrained_model_dir="",

      # ========== 多任务学习 (MultiProblem) ==========
      # 用于两种情况的阈值：
      # 1. 固定混合调度中主任务的概率
      # 2. 指数调度中混合停止时的限制（如 0.5 表示在 50-50 混合时停止）
      multiproblem_schedule_threshold=0.5,
      # 多任务（超过 2 个）的每任务阈值字符串（浮点数列表，逗号分隔）
      # 这些数值会被归一化为各任务的概率
      multiproblem_per_task_threshold="",
      # 混合数据集比例达到 threshold 时的样本数
      multiproblem_schedule_max_examples=1e7,
      # 多任务数据混合调度策略，如 "constant"（固定比例混合）
      multiproblem_mixing_schedule="constant",
      # 是否对分类问题的输入序列损失和目标标签损失重新加权
      multiproblem_reweight_label_loss=False,
      # 分类问题中目标标签的损失权重（输入部分权重 = 1 - 此值）
      multiproblem_label_weight=0.5,

      # ========== 相对位置编码 ==========
      # 相对位置编码的最大距离（超出此距离的位置共用同一个 embedding）
      # 0 表示不使用相对位置编码
      max_relative_position=0,
      # 多个注意力头是否共享同一套相对位置 embedding
      heads_share_relative_embedding=False,
      # 是否也将相对位置 embedding 加到 value 上（而不只是 query-key 的点积上）
      add_relative_to_values=False,

      # ========== TPU 专用 ==========
      # 是否启用在每个训练步骤上执行的 host_call。
      # 如果 host_call 函数较慢，可能导致性能下降（无法跟上 TPU 侧的计算速度）。
      tpu_enable_host_call=False,
      # 将批次维度填充到 batch_multiple 的最近倍数
      pad_batch=False,
      # 多任务时，是否跳过语言模型数据的评估（语言模型评估可能很耗时）
      # 如果设为 False，请将 eval_steps 设置为较大值（如 6000 或 10000）
      multiproblem_target_eval_only=False,
      # 将词表大小扩展到 2 的幂次方（为新任务 ID 和标签类别预留空间）
      multiproblem_vocab_size=-1,
      # 在多任务生成任务中，拼接输入和目标前需要手动截断的最大长度
      multiproblem_max_input_length=-1,
      multiproblem_max_target_length=-1,
      # 如果为正，使 MultiProblem 中的训练目标具有固定长度
      multiproblem_fixed_train_length=-1,
      # 从第二个模型加载权重（用于预训练时分别初始化编码器和解码器）
      warm_start_from_second="",

      # ========== 区域注意力 (Area Attention) ==========
      # 区域注意力是一种计算效率更高的注意力机制，
      # 对输入区域（矩形区域）而非单个位置计算注意力
      area_value_mode="none",  # 区域注意力 value 的处理方式
      area_key_mode="none",    # 区域注意力 key 的处理方式
      # 从底部开始使用区域注意力的层数（0 表示全部使用普通注意力）
      num_area_layers=0,
      # 区域的最大宽度
      max_area_width=1,
      # 区域的最大高度
      max_area_height=1,
      # 内存高度
      memory_height=1,
      # 是否使用 GPU 自动混合精度（通过图重写实现）
      gpu_automatic_mixed_precision=False,
  )


class RangedHParams(object):
  """定义用于超参数搜索/调优的参数范围。
  
  在进行超参数调优（如 Google Cloud ML Engine 的自动调参服务）时，
  需要为每个超参数指定一个搜索范围。
  
  此类支持四种类型的超参数：
  - categorical（分类型）：从一组离散的类别值中选择
  - discrete（离散型）：从一组离散的数值中选择
  - float（浮点型）：在一个连续范围内搜索浮点数
  - int（整型）：在一个连续范围内搜索整数
  
  同时支持三种搜索空间的缩放方式：
  - LINEAR_SCALE：线性缩放，均匀搜索
  - LOG_SCALE：对数缩放，对小值搜索更密集（适合学习率等超参数）
  - REVERSE_LOG_SCALE：反对数缩放
  """

  # 从 ParameterConfig proto 定义的缩放类型常量
  LINEAR_SCALE = 1       # 线性缩放
  LOG_SCALE = 2          # 对数缩放（适合学习率等量级差异大的参数）
  REVERSE_LOG_SCALE = 3  # 反对数缩放

  # 缩放类型名称映射（用于转换成 Cloud ML Engine 配置格式）
  SCALES_STR = {
      LINEAR_SCALE: "UNIT_LINEAR_SCALE",
      LOG_SCALE: "UNIT_LOG_SCALE",
      REVERSE_LOG_SCALE: "UNIT_REVERSE_LOG_SCALE",
  }

  def __init__(self):
    """初始化各类型超参数的存储字典。"""
    # 分类型超参数字典：{参数名 -> (名称, 类别列表, 长度)}
    self._categorical_params = {}
    # 离散型超参数字典：{参数名 -> (名称, 可行点列表, 缩放类型, 长度)}
    self._discrete_params = {}
    # 浮点型超参数字典：{参数名 -> (名称, 最小值, 最大值, 缩放类型, 长度)}
    self._float_params = {}
    # 整型超参数字典：{参数名 -> (名称, 最小值, 最大值, 缩放类型, 长度)}
    self._int_params = {}

  def _check_reset_and_type_change(self, name, orig_ctr):
    """检查是否有重复注册或类型变更的情况。
    
    Args:
      name: 超参数名称
      orig_ctr: 超参数本应存储在的字典（其类型的字典）
      
    Raises:
      ValueError: 如果同名超参数之前用不同类型注册过
    """
    # 如果同名参数已经在相同类型字典中存在，发出警告（允许覆盖）
    if name in orig_ctr:
      tf.logging.warning("Overwriting hparam %s", name)

    # 构建所有类型字典及其名称的列表，用于检测类型冲突
    ctr_names = [
        (self._categorical_params, "categorical"),
        (self._discrete_params, "discrete"),
        (self._float_params, "float"),
        (self._int_params, "int"),
    ]
    # 解压出字典列表和对应名称列表
    ctrs, names = list(zip(*ctr_names))
    # 获取当前传入字典对应的类型名称
    orig_name = names[ctrs.index(orig_ctr)]

    # 检查同名参数是否在其他类型字典中已存在
    for ctr, ctr_name in ctr_names:
      if ctr is orig_ctr:
        continue  # 跳过当前类型字典（允许同类型覆盖）

      # 发现在其他类型字典中存在同名参数，抛出类型冲突异常
      if name in ctr:
        raise ValueError("Setting hyperparameter %s as type %s, but a "
                         "hyperparemeter of the same name was originally "
                         "registered as type %s" % (name, ctr_name, orig_name))

  def set_categorical(self, name, categories, length=None):
    """注册一个分类型超参数及其候选值列表。
    
    Args:
      name: 超参数名称（与 HParams 中的参数名对应）
      categories: 候选值列表，如 ["adam", "sgd", "adagrad"]
      length: 可选，超参数向量的长度（用于向量型参数）
    """
    self._check_reset_and_type_change(name, self._categorical_params)
    self._categorical_params[name] = (name, categories, length)

  def set_discrete(self, name, feasible_points, scale=None, length=None):
    """注册一个离散数值型超参数及其候选点集合。
    
    Args:
      name: 超参数名称
      feasible_points: 候选数值列表，如 [1, 2, 4, 8, 16]
      scale: 搜索空间的缩放类型（LINEAR_SCALE/LOG_SCALE/REVERSE_LOG_SCALE）
      length: 可选，超参数向量的长度
    """
    self._check_reset_and_type_change(name, self._discrete_params)
    self._discrete_params[name] = (name, feasible_points, scale, length)

  def set_float(self, name, min_val, max_val, scale=None, length=None):
    """注册一个浮点型超参数及其取值范围。
    
    Args:
      name: 超参数名称
      min_val: 最小值（含）
      max_val: 最大值（含）
      scale: 搜索空间的缩放类型（建议学习率等使用 LOG_SCALE）
      length: 可选，超参数向量的长度
    """
    self._check_reset_and_type_change(name, self._float_params)
    self._float_params[name] = (name, min_val, max_val, scale, length)

  def set_int(self, name, min_val, max_val, scale=None, length=None):
    """注册一个整型超参数及其取值范围。
    
    Args:
      name: 超参数名称
      min_val: 最小整数值（含）
      max_val: 最大整数值（含）
      scale: 搜索空间的缩放类型
      length: 可选，超参数向量的长度
    """
    self._check_reset_and_type_change(name, self._int_params)
    self._int_params[name] = (name, min_val, max_val, scale, length)

  def fix_select_params(self, hp):
    """将 HParams 中的参数值固定（从范围缩小到单个点），用于部分参数固定的调优场景。
    
    Args:
      hp: HParams 对象，其中的参数值将被固定为当前值（单点离散范围）
    """
    # 获取所有超参数字典
    ctrs = [
        self._categorical_params, self._discrete_params, self._float_params,
        self._int_params
    ]
    # 遍历 hp 中的每个参数
    for key, val in hp.values().iteritems():
      # 从所有类型字典中删除该参数（如果存在）
      for ctr in ctrs:
        if key in ctr:
          del ctr[key]
      # 将该参数注册为只有当前值的离散型参数（固定值）
      self.set_discrete(key, [val])

  def to_parameter_specs(self, name_prefix=""):
    """将超参数范围转换为适合 Google Cloud ML Engine 超参数调优的格式。
    
    Cloud ML Engine 需要以特定 JSON 格式描述超参数搜索空间，
    此方法将内部表示转换为相应的字典列表。
    
    Args:
      name_prefix: 超参数名称前缀（用于区分嵌套配置）
      
    Returns:
      适合 Cloud ML Engine 超参数调优 API 的参数规格列表（字典列表）
    """
    specs = []
    
    # 处理分类型参数
    for name, categories, _ in self._categorical_params.values():
      spec = {
          "parameterName": name_prefix + name,  # 参数名称（含前缀）
          "type": "CATEGORICAL",                 # 参数类型
          "categoricalValues": categories,       # 候选类别值列表
      }
      specs.append(spec)

    # 处理离散数值型参数
    for name, feasible_points, scale, _ in self._discrete_params.values():
      spec = {
          "parameterName": name_prefix + name,
          "type": "DISCRETE",
          "discreteValues": feasible_points,  # 离散候选点列表
      }
      if scale:
        spec["scaleType"] = self.SCALES_STR[scale]  # 添加缩放类型
      specs.append(spec)

    # 处理浮点型参数
    for name, min_val, max_val, scale, _ in self._float_params.values():
      spec = {
          "parameterName": name_prefix + name,
          "type": "DOUBLE",
          "minValue": min_val,  # 最小值
          "maxValue": max_val,  # 最大值
      }
      if scale:
        spec["scaleType"] = self.SCALES_STR[scale]
      specs.append(spec)

    # 处理整型参数
    for name, min_val, max_val, scale, _ in self._int_params.values():
      spec = {
          "parameterName": name_prefix + name,
          "type": "INTEGER",
          "minValue": min_val,
          "maxValue": max_val,
      }
      if scale:
        spec["scaleType"] = self.SCALES_STR[scale]
      specs.append(spec)

    return specs


@registry.register_ranged_hparams("basic1")  # 注册为名称 "basic1" 的超参数范围配置
def basic_range1(ranged_hparams):
  """定义基础超参数的搜索范围，用于自动超参数调优。
  
  这些范围覆盖了模型结构和训练的核心超参数，
  可以用于 Google Cloud ML Engine 或其他超参数搜索框架。
  
  Args:
    ranged_hparams: RangedHParams 对象，用于设置各参数的搜索范围
  """
  rhp = ranged_hparams  # 简写别名
  
  # 批次大小：从 3 个离散值中选择
  rhp.set_discrete("batch_size", [1024, 2048, 4096])
  # 隐藏层数量：1 到 6 层
  rhp.set_discrete("num_hidden_layers", [1, 2, 3, 4, 5, 6])
  # 隐藏层维度：对数刻度搜索（小值更密集，适合此类参数）
  rhp.set_discrete("hidden_size", [32, 64, 128, 256, 512], scale=rhp.LOG_SCALE)
  # 卷积核尺寸
  rhp.set_discrete("kernel_height", [1, 3, 5, 7])
  rhp.set_discrete("kernel_width", [1, 3, 5, 7])
  # 压缩步骤数
  rhp.set_discrete("compress_steps", [0, 1, 2])
  # Dropout 概率：0.0 到 0.5 的连续范围
  rhp.set_float("dropout", 0.0, 0.5)
  # 权重衰减：对数刻度，1e-4 到 10
  rhp.set_float("weight_decay", 1e-4, 10.0, scale=rhp.LOG_SCALE)
  # 标签平滑系数：0.0 到 0.2
  rhp.set_float("label_smoothing", 0.0, 0.2)
  # 梯度裁剪范数：对数刻度，0.01 到 50
  rhp.set_float("clip_grad_norm", 0.01, 50.0, scale=rhp.LOG_SCALE)
  # 学习率：对数刻度，0.005 到 2.0
  rhp.set_float("learning_rate", 0.005, 2.0, scale=rhp.LOG_SCALE)
  # 初始化方式：从三种常见方式中选择
  rhp.set_categorical("initializer",
                      ["uniform", "orthogonal", "uniform_unit_scaling"])
  # 初始化增益
  rhp.set_float("initializer_gain", 0.5, 3.5)
  # 学习率衰减方案
  rhp.set_categorical("learning_rate_decay_scheme",
                      ["none", "sqrt", "noam", "exp"])
  # Adam 优化器参数
  rhp.set_float("optimizer_adam_epsilon", 1e-7, 1e-2, scale=rhp.LOG_SCALE)
  rhp.set_float("optimizer_adam_beta1", 0.8, 0.9)
  rhp.set_float("optimizer_adam_beta2", 0.995, 0.999)
  # 优化器类型
  rhp.set_categorical(
      "optimizer",
      ["adam", "adagrad", "momentum", "rms_prop", "sgd", "yellow_fin"])


@registry.register_ranged_hparams  # 使用函数名作为注册名称
def basic_moe_range(rhp):
  """MoE (混合专家模型) 超参数的调优范围。
  
  当此参数未使用时，保留它可以让我们观察由参数初始化引入的方差（随机性）。
  
  Args:
    rhp: RangedHParams 对象
  """
  # MoE 负载均衡损失系数：在 0.01 到 0.02 的小范围内搜索
  rhp.set_float("moe_loss_coef", 0.01, 0.02)
