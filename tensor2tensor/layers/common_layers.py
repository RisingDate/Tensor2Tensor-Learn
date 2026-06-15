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

"""多模型通用的神经网络层（Layers）实现。

本文件是 Tensor2Tensor 框架的核心工具库，提供了构建深度学习模型所需的
各种基础组件，包括：
- 卷积（Conv）和全连接（Dense）层的封装
- 归一化（Normalization）：层归一化、批归一化等
- 激活函数：ReLU、GELU、SRU 等
- Dropout 和正则化
- 序列处理工具：padding、masking、位置编码
- 损失函数：交叉熵、离散混合 logistic 损失
- 内存高效的反向传播技术
- GAN（生成对抗网络）相关工具
- 集合（Set）处理层
- 视频处理工具
等等

这些基础函数被 Transformer、ResNet、LSTM 等多种模型共享使用。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections  # Python 内置集合模块（OrderedDict、namedtuple等）
import contextlib   # 上下文管理器工具
import functools    # 高阶函数工具（如 partial、wraps）
import math         # 数学函数

from absl import logging  # Google 的 abseil 日志库
import numpy as np        # 数值计算库
from six.moves import range  # Python 2/3 兼容的 range 函数 # pylint: disable=redefined-builtin

from tensor2tensor.utils import contrib  # TF Contrib 模块的兼容封装
import tensorflow.compat.v1 as tf        # TensorFlow 1.x 兼容 API
import tensorflow_probability as tfp     # TensorFlow Probability（概率模型库）

from tensorflow.python.framework import function       # TF 自定义函数框架
from tensorflow.python.framework import ops            # TF 操作基础框架
from tensorflow.python.ops import control_flow_util    # 控制流工具（XLA 检测等）
from tensorflow.python.ops import inplace_ops          # 原地操作工具


# TODO(lukaszkaiser): remove this function when not needed any more.
def layers():
  """获取适配 TF 1.x 和 TF 2.x 的 layers 模块（临时兼容方案）。
  
  TensorFlow 在 2.x 版本中将 tf.layers 迁移到了 tf.keras.layers。
  此函数通过检测 TF 版本来返回正确的 layers 模块，确保代码向前兼容。
  
  Returns:
    layers_module: tf.layers（TF1）或 tf.keras.layers（TF2）
  """
  layers_module = None
  try:
    # 尝试获取 TF1 的 layers 模块
    layers_module = tf.layers
  except AttributeError:
    logging.info("Cannot access tf.layers, trying TF2 layers.")
  try:
    from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
    if tf2.enabled():
      logging.info("Running in V2 mode, using Keras layers.")
      # TF2 模式下使用 Keras layers
      layers_module = tf.keras.layers
  except ImportError:
    pass
  return layers_module


# @function.Defun 是一个装饰器，将 Python 函数注册为 TensorFlow 的自定义函数（op）
# python_grad_func: 指定 Python 级别的梯度函数
#   lambda x, dy: tf.convert_to_tensor(dy) 表示将梯度 dy 从 IndexedSlices 转换为稠密 Tensor
# shape_func: 指定输出形状推导函数
#   lambda op: [op.inputs[0].get_shape()] 表示输出形状与输入相同
@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """恒等操作，但将其梯度从稀疏格式（IndexedSlices）转换为稠密 Tensor。

  使用场景：
  当 tf.concat 的输出最终被 tf.gather 使用时，梯度会是稀疏的 IndexedSlices。
  TensorFlow 对稀疏梯度没有 GPU 实现，这会强制梯度计算在 CPU 上进行，导致性能下降。
  
  解决方案：将 tf.concat(x) 替换为 convert_gradient_to_tensor(tf.concat(x))，
  这样反向传播时梯度会被转换为稠密 Tensor，从而使用更高效的 GPU 实现。
  
  注意：
  - IndexedSlices 是稀疏张量的表示，只存储非零值及其索引
  - 转换为稠密 Tensor 会增加内存使用，但减少 GPU/CPU 之间的数据传输

  Args:
    x: 输入张量（Tensor）

  Returns:
    与输入完全相同的张量（但梯度流不同）
  """
  return x


def is_xla_compiled():
  """检查当前计算图是否将被 XLA 编译器编译。

  XLA（Accelerated Linear Algebra）是 TensorFlow 的即时编译器，
  可以将计算图编译为高效的机器代码，特别适合 TPU 加速。
  
  如果此函数返回 True，模型代码需要确保：
  1. 所有使用的 op 都有 XLA 实现
  2. 所有张量形状在编译时可以静态确定（不能有动态形状）

  Returns:
    bool: 如果当前在 XLA 编译上下文中返回 True，否则返回 False
  """
  # 获取当前计算图的控制流上下文
  ctxt = tf.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  # 检查是否处于 XLA 上下文中
  return control_flow_util.GetContainingXLAContext(ctxt) is not None


def to_float(x):
  """将张量转换为 float32 类型（替代已弃用的 tf.to_float）。
  
  Args:
    x: 任意类型的张量（通常是 int32 或 bool）
    
  Returns:
    float32 类型的张量
  """
  return tf.cast(x, tf.float32)


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
  """与 tf.nn.dropout 相同，但使用 broadcast_dims 参数代替 noise_shape。

  标准 tf.nn.dropout 使用 noise_shape 来指定随机 mask 的形状。
  此函数改用 broadcast_dims（广播维度列表），更直观：
  指定在哪些维度上共享同一个 dropout mask（该维度的 noise_shape=1）。
  
  使用场景：
  - 设置 broadcast_dims=[0] 表示在 batch 维度上共享 dropout mask，
    同一批次内的所有样本使用相同的 mask（节省内存）
  - 设置 broadcast_dims=[1] 表示在序列长度维度上共享 mask，
    相当于对整个序列的某些特征维度一起丢弃
  
  这对于 Transformer 的层预处理/后处理中使用的 dropout 特别有用。

  Args:
    x: 浮点型张量，被 dropout 的输入
    keep_prob: 标量张量，每个元素被保留的概率（例如 0.9 表示 10% dropout）
    broadcast_dims: 可选的整数列表，指定在哪些维度上广播 dropout mask
                   （这些维度的 noise_shape=1），支持负索引
    **kwargs: 其他传递给 tf.nn.dropout 的关键字参数（不含 noise_shape）

  Returns:
    与 x 形状相同的张量，dropout 后的结果
  """
  assert "noise_shape" not in kwargs  # 不允许同时指定 noise_shape
  if broadcast_dims:
    shape = tf.shape(x)  # 获取动态形状
    ndims = len(x.get_shape())  # 获取静态维度数
    # 支持负索引（如 -1 表示最后一维）
    broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
    # 构建 noise_shape：广播维度的大小设为 1，其他维度保持原大小
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in range(ndims)
    ]
  return tf.nn.dropout(x, keep_prob, **kwargs)


def comma_separated_string_to_integer_list(s):
  """将逗号分隔的字符串转换为整数列表。
  
  例如："1,2,3" -> [1, 2, 3]，"" -> []
  
  Args:
    s: 逗号分隔的整数字符串
    
  Returns:
    整数列表
  """
  return [int(i) for i in s.split(",") if i]


def saturating_sigmoid(x):
  """饱和 sigmoid 函数：1.2 * sigmoid(x) - 0.1，结果截断到 [0, 1]。
  
  这是标准 sigmoid 的改进版本，通过拉伸曲线使其在接近 0 和 1 时更"硬"
  （更快达到饱和状态），同时在中间区域（0.1 到 0.9 之间）
  比普通 sigmoid 有更好的梯度传播。
  
  原理：
  - 普通 sigmoid 在输出 0 和 1 附近梯度趋近于 0（梯度消失）
  - 饱和 sigmoid 通过缩放让更多输入区间对应 0 和 1 的输出，
    但在中间区域仍保持较大梯度
  """
  with tf.name_scope("saturating_sigmoid", values=[x]):
    y = tf.sigmoid(x)
    # 拉伸并截断：将 sigmoid 的 [0,1] 输出映射到稍宽的范围，再截断
    return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))


def hard_sigmoid(x, saturation_limit=0.9):
  """硬 sigmoid 函数，带饱和惩罚项。
  
  硬 sigmoid 是 sigmoid 的分段线性近似，计算效率更高。
  同时计算饱和惩罚：当 |x| > saturation_limit 时增加惩罚，
  防止激活值过大导致饱和。
  
  Args:
    x: 输入张量
    saturation_limit: 饱和阈值，超过此绝对值时增加惩罚
    
  Returns:
    (output, saturation_cost): 函数输出值和饱和惩罚损失
  """
  # 计算饱和惩罚：超出 saturation_limit 的部分取均值作为额外损失
  saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
  # 将 x 从 [-1,1] 线性映射到 [0,1]，然后用 ReLU 截断负值，minimum 截断大于1的值
  x_shifted = 0.5 * x + 0.5
  return tf.minimum(1.0, tf.nn.relu(x_shifted)), saturation_cost


def hard_tanh(x, saturation_limit=0.9):
  """硬 tanh 函数，带饱和惩罚项。
  
  硬 tanh 将输出截断到 [-1, 1]，同时对过大的输入值施加惩罚。
  
  Args:
    x: 输入张量
    saturation_limit: 饱和阈值
    
  Returns:
    (output, saturation_cost): 截断到 [-1,1] 的输出和饱和惩罚损失
  """
  saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
  return tf.minimum(1.0, tf.maximum(x, -1.0)), saturation_cost


def inverse_exp_decay(max_step, min_value=0.01, step=None):
  """逆指数衰减：从 min_value 指数增长到 1.0，在 max_step 步时达到 1.0。
  
  这是一种"反向"衰减曲线，通常用于学习率预热（warmup）：
  - 在训练开始（step=0）时，值为 min_value（如 0.01）
  - 随着训练步数增加，值指数增长
  - 在 step=max_step 时，值恰好达到 1.0
  
  数学公式：y = base^(max_step - step)
  其中 base = exp(log(min_value) / max_step) = min_value^(1/max_step)
  
  Args:
    max_step: 值增长到 1.0 时的训练步数
    min_value: 起始值（step=0 时的值）
    step: 当前训练步数，None 时从全局步数获取
    
  Returns:
    当前步数对应的值（在 [min_value, 1.0] 范围内）
  """
  # 计算基数：base^max_step = min_value => base = min_value^(1/max_step)
  inv_base = tf.exp(tf.log(min_value) / float(max_step))
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0  # 无法获取步数时，直接返回 1.0
  step = to_float(step)
  # 当 step >= max_step 时，指数为 0，返回 base^0 = 1.0
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01, step=None):
  """逆线性衰减：从 min_value 线性增长到 1.0，在 max_step 步时达到 1.0。
  
  与 inverse_exp_decay 类似，但增长曲线是线性的（直线）。
  常用于计划采样（scheduled sampling）的概率预热。
  
  数学公式：y = progress * (1 - min_value) + min_value
  其中 progress = min(step / max_step, 1.0)
  
  Args:
    max_step: 值增长到 1.0 时的步数
    min_value: 起始值
    step: 当前训练步数
    
  Returns:
    当前步数对应的值（在 [min_value, 1.0] 范围内）
  """
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)
  # progress: 训练进度，从 0 到 1（max_step 后保持 1.0）
  progress = tf.minimum(step / float(max_step), 1.0)
  # 线性插值：从 min_value 到 1.0
  return progress * (1.0 - min_value) + min_value


def inverse_sigmoid_decay(max_step, min_value=0.01, step=None):
  """逆 sigmoid 衰减：从 min_value 以 S 形曲线增长到 1.0，在 max_step 步时达到约 1.0。
  
  S 形增长曲线的特点：
  - 初期增长缓慢（平稳起步）
  - 中期快速增长（加速阶段）
  - 后期再次减缓（平稳收尾）
  
  这比线性预热更自然，适合某些需要平滑过渡的场景。
  
  Args:
    max_step: 值接近 1.0 时的步数
    min_value: 起始值（必须 > 0 且 < 0.5）
    step: 当前训练步数
    
  Returns:
    当前步数对应的值（在 [min_value, 1.0] 范围内）
  """
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)

  def sigmoid(x):
    """标准 sigmoid 函数：1 / (1 + e^(-x))"""
    return 1 / (1 + tf.exp(-x))

  def inv_sigmoid(y):
    """sigmoid 的逆函数（logit 函数）：log(y / (1-y))"""
    return tf.log(y / (1 - y))

  assert min_value > 0, (
      "sigmoid's output is always >0 and <1. min_value must respect "
      "these bounds for interpolation to work.")
  assert min_value < 0.5, "Must choose min_value on the left half of sigmoid."

  # 找到使 sigmoid(x_min) = y_min 和 sigmoid(x_max) = y_max 的 x 值
  # 将训练步数 [0, max_step] 映射到 sigmoid 的 [x_min, x_max]
  y_min = min_value
  y_max = 1.0 - min_value
  x_min = inv_sigmoid(y_min)  # sigmoid 输入的最小值（对应 y_min）
  x_max = inv_sigmoid(y_max)  # sigmoid 输入的最大值（对应 y_max）

  x = tf.minimum(step / float(max_step), 1.0)  # 将步数归一化到 [0, 1]
  x = x_min + (x_max - x_min) * x  # 线性映射到 [x_min, x_max]
  y = sigmoid(x)  # 通过 sigmoid 得到 [y_min, y_max]

  # 将 [y_min, y_max] 重新归一化并缩放到 [y_min, 1.0]
  y = (y - y_min) / (y_max - y_min)  # 归一化到 [0, 1]
  y = y * (1.0 - y_min)  # 缩放到 [0, 1-y_min]
  y += y_min  # 平移到 [y_min, 1]
  return y


def shakeshake2_py(x, y, equal=False, individual=False):
  """Shake-Shake 正则化：对两个张量的随机加权求和（Python 版本）。
  
  Shake-Shake（论文 "Shake-Shake regularization"）是一种针对多分支神经网络的正则化方法：
  - 前向传播时，用随机权重 alpha 和 1-alpha 混合两个分支的输出
  - 反向传播时，用另一组独立随机权重
  这种不一致性迫使每个分支独立学习，类似于 Dropout 的集成效果。
  
  Args:
    x: 第一个张量
    y: 第二个张量（形状与 x 相同）
    equal: 如果 True，使用固定权重 alpha=0.5（均等混合，用于推理/测试阶段）
    individual: 如果 True，为 batch 中的每个样本独立采样一个 alpha
    
  Returns:
    alpha * x + (1 - alpha) * y，其中 alpha 根据参数策略决定
  """
  if equal:
    alpha = 0.5
  elif individual:
    alpha = tf.random_uniform(tf.get_shape(x)[:1])
  else:
    alpha = tf.random_uniform([])

  return alpha * x + (1.0 - alpha) * y


@function.Defun()
def shakeshake2_grad(x1, x2, dy):
  """覆盖 shakeshake2 的梯度计算（反向传播时使用不同的随机 alpha）。
  
  前向传播用一个随机 alpha，反向传播用另一个随机 alpha，
  这种不一致性是 Shake-Shake 正则化的关键。
  """
  y = shakeshake2_py(x1, x2)
  dx = tf.gradients(ys=[y], xs=[x1, x2], grad_ys=[dy])
  return dx


@function.Defun()
def shakeshake2_indiv_grad(x1, x2, dy):
  """shakeshake2_indiv 的梯度：每个样本独立采样随机 alpha。"""
  y = shakeshake2_py(x1, x2, individual=True)
  dx = tf.gradients(ys=[y], xs=[x1, x2], grad_ys=[dy])
  return dx


@function.Defun()
def shakeshake2_equal_grad(x1, x2, dy):
  """shakeshake2_eqgrad 的梯度：前向和反向均使用 alpha=0.5（均等梯度版本）。"""
  y = shakeshake2_py(x1, x2, equal=True)
  dx = tf.gradients(ys=[y], xs=[x1, x2], grad_ys=[dy])
  return dx


@function.Defun(grad_func=shakeshake2_grad)
def shakeshake2(x1, x2):
  """标准 Shake-Shake：前向和反向使用不同的随机权重 alpha。
  
  前向：随机 alpha（全局标量，所有样本共享）
  反向：另一个随机 alpha（由 shakeshake2_grad 实现）
  """
  return shakeshake2_py(x1, x2)


@function.Defun(grad_func=shakeshake2_indiv_grad)
def shakeshake2_indiv(x1, x2):
  """Individual Shake-Shake：每个样本使用独立的随机权重 alpha。"""
  return shakeshake2_py(x1, x2, individual=True)


@function.Defun(grad_func=shakeshake2_equal_grad)
def shakeshake2_eqgrad(x1, x2):
  """Equal-grad Shake-Shake：前向用随机 alpha，反向用均等 alpha=0.5。"""
  return shakeshake2_py(x1, x2)


def shakeshake(xs, equal_grad=False):
  """多分支 Shake-Shake 正则化（目前通过两两配对来近似）。
  
  递归地将多个张量两两进行 shake-shake 融合：
  - 将列表分成两半，分别递归融合
  - 再对结果进行一次 shake-shake
  
  Args:
    xs: 张量列表（多个网络分支的输出）
    equal_grad: 是否使用均等梯度（equal_grad=True 用于测试/推理）
    
  Returns:
    所有分支融合后的张量
  """
  if len(xs) == 1:
    return xs[0]
  div = (len(xs) + 1) // 2
  arg1 = shakeshake(xs[:div], equal_grad=equal_grad)
  arg2 = shakeshake(xs[div:], equal_grad=equal_grad)
  if equal_grad:
    return shakeshake2_eqgrad(arg1, arg2)
  return shakeshake2(arg1, arg2)


def convert_rgb_to_real(x):
  """将 RGB 像素值（0-255 整数）转换为 [0, 1] 范围的浮点数。
  
  用于图像预处理：将 uint8 格式的像素值归一化到 [0, 1]。
  """
  with tf.name_scope("rgb_to_real", values=[x]):
    x = to_float(x)
    x /= 255.0
    return x


def convert_rgb_to_symmetric_real(x):
  """将 RGB 像素值（0-255 整数）转换为 [-1, 1] 范围的浮点数。
  
  用于 GAN 等模型的图像预处理：将像素值对称归一化到 [-1, 1]。
  公式：(x / 127.5) - 1
  """
  with tf.name_scope("rgb_to_real", values=[x]):
    x = to_float(x)
    # Convert each pixel intensity in [0, 1, 2, ..., 255] into a real number in
    # the range [-1, 1].
    x = (x / 127.5) - 1
    return x


def convert_real_to_rgb(x):
  """将 [0, 1] 范围的浮点数转换回 RGB 像素值（乘以 255）。
  
  与 convert_rgb_to_real 的逆操作，用于生成图像的后处理。
  """
  with tf.name_scope("real_to_rgb", values=[x]):
    x *= 255.0
    return x


def expand_squeeze_to_nd(x, n, squeeze_dim=2, expand_dim=-1):
  """通过 squeeze 和 expand_dims 将张量调整为 n 维。
  
  如果当前维度 > n，则在 squeeze_dim 维度上压缩（去掉大小为 1 的维度）；
  如果当前维度 < n，则在 expand_dim 维度上扩展（增加大小为 1 的维度）。
  
  Args:
    x: 输入张量
    n: 目标维度数
    squeeze_dim: 要压缩的维度（当维度过多时）
    expand_dim: 要扩展的维度（当维度不足时）
    
  Returns:
    n 维张量
  """
  if len(x.shape) > n:
    while len(x.shape) != n:
      x = tf.squeeze(x, [squeeze_dim])
  else:
    while len(x.shape) != n:
      x = tf.expand_dims(x, expand_dim)
  return x


def standardize_images(x):
  """对批量图像或视频进行标准化处理（零均值、单位方差）。
  
  对每张图像独立地减去均值并除以标准差，使像素值中心化且方差为 1。
  这是图像分类任务中常见的预处理步骤，有助于加速训练收敛。
  
  公式：(x - mean(x)) / max(std(x), 1/sqrt(num_pixels))
  分母取最大值是为了防止除以接近零的标准差。
  """
  with tf.name_scope("standardize_images", values=[x]):
    x_shape = shape_list(x)
    x = to_float(tf.reshape(x, [-1] + x_shape[-3:]))
    x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_variance = tf.reduce_mean(
        tf.squared_difference(x, x_mean), axis=[1, 2], keepdims=True)
    num_pixels = to_float(x_shape[-2] * x_shape[-3])
    x = (x - x_mean) / tf.maximum(tf.sqrt(x_variance), tf.rsqrt(num_pixels))
    return tf.reshape(x, x_shape)


def flatten4d3d(x):
  """将 4D 张量 [batch, height, width, depth] 展平为 3D 张量 [batch, height*width, depth]。
  
  用于将 2D 空间特征图（图像特征）转换为 1D 序列形式，
  以便输入到序列模型（如 Transformer）中。
  """
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result


# TODO(noam): remove this function after TPUs do gather faster.
def gather(params, indices, dtype=tf.float32):
  """gather 操作的 TPU 优化版本，在 TPU 上比 tf.gather 更快。
  
  在 CPU/GPU 上直接使用 tf.gather（标准实现）。
  在 TPU 上，tf.gather 较慢，改用 one-hot 编码乘以参数矩阵的方式实现：
    result = one_hot(indices) @ params
  这种方式可以利用 TPU 优化的矩阵乘法。
  
  Args:
    params: 参数矩阵（如 embedding 矩阵），形状 [vocab_size, embed_dim]
    indices: 整数索引张量，形状任意
    dtype: 输出数据类型
    
  Returns:
    从 params 中按 indices 查找到的结果
  """
  if not is_xla_compiled():
    # 非 XLA/TPU 环境：直接使用标准 tf.gather
    return tf.gather(params, indices)
  # TPU 优化实现：通过 one-hot 向量与参数矩阵的矩阵乘法来模拟 gather
  vocab_size = params.get_shape().as_list()[0]
  indices_flat = tf.reshape(indices, [-1])  # 将索引展平为 1D
  # one_hot(indices_flat, vocab_size) @ params 等价于按索引查找
  out = tf.matmul(tf.one_hot(indices_flat, vocab_size, dtype=dtype), params)
  # 将结果重新 reshape 成与 indices 对应的形状（最后加一维）
  out = reshape_like(out, tf.expand_dims(indices, -1))
  return out


# TODO(noam): remove this function after TPUs do cumsum faster.
def cumsum(x, axis=0, exclusive=False):
  """累积求和的 TPU 优化版本。
  
  在 CPU/GPU 上等价于 tf.cumsum。
  在 TPU 上（截至 2018 年 4 月），通过三角矩阵掩码实现的矩阵乘法比原生 tf.cumsum 更快
  （当轴维度不太大时）。
  
  原理：
  - 构建一个下三角矩阵 mask（mask[i][j] = 1 if j <= i else 0）
  - 通过 tensordot 实现 cumsum：ret[..., i, ...] = sum(x[..., j, ...] for j <= i)
  
  exclusive=True 时计算不包含当前元素的前缀和（exclusive cumsum），即：
    ret[i] = sum(x[j] for j < i)
  
  Args:
    x: 输入张量
    axis: 进行累积求和的轴
    exclusive: 是否计算独占前缀和（不含当前元素）
    
  Returns:
    与 x 形状相同的累积求和结果
  """
  if not is_xla_compiled():
    # 非 TPU 环境：直接使用标准 tf.cumsum
    return tf.cumsum(x, axis=axis, exclusive=exclusive)
  # TPU 优化实现：通过掩码矩阵乘法实现
  x_shape = shape_list(x)
  rank = len(x_shape)
  length = x_shape[axis]
  my_range = tf.range(length)  # [0, 1, 2, ..., length-1]
  # exclusive=True 时使用 <（不含当前），False 时使用 <=（含当前）
  comparator = tf.less if exclusive else tf.less_equal
  # 构建下三角（或严格下三角）掩码矩阵
  mask = tf.cast(
      comparator(tf.expand_dims(my_range, 1), tf.expand_dims(my_range, 0)),
      x.dtype)
  # 通过 tensordot 实现矩阵乘法：沿 axis 维度进行累积求和
  ret = tf.tensordot(x, mask, axes=[[axis], [0]])
  if axis != rank - 1:
    # 如果 axis 不是最后一维，需要转置回原来的维度顺序
    ret = tf.transpose(
        ret,
        list(range(axis)) + [rank - 1] + list(range(axis, rank - 1)))
  return ret


def dropout_no_scaling(x, keep_prob):
  """不带缩放的 Dropout，也适用于整型张量。
  
  标准 tf.nn.dropout 在丢弃元素后会将剩余元素乘以 1/keep_prob（期望缩放），
  这样可以保持期望值不变。但此函数不做缩放，直接将被丢弃的位置置零。
  
  用途：用于 token-level dropout（符号 dropout），直接将 token 置零而不需要缩放。
  
  Args:
    x: 输入张量（整型或浮点型均可）
    keep_prob: 每个元素被保留的概率（1.0 表示不丢弃）
    
  Returns:
    与 x 形状相同的张量，某些位置被置零（不缩放）
  """
  if keep_prob == 1.0:
    return x  # keep_prob=1 时不做任何处理，直接返回
  # 生成随机掩码：keep_prob 概率为 True（保留），1-keep_prob 概率为 False（丢弃）
  mask = tf.less(tf.random_uniform(tf.shape(x)), keep_prob)
  # 将掩码转换成与 x 相同的类型，并与 x 相乘（False=0 相当于丢弃）
  return x * cast_like(mask, x)


def embedding(x,
              vocab_size,
              dense_size,
              name=None,
              reuse=None,
              multiplier=1.0,
              symbol_dropout_rate=0.0,
              embedding_var=None,
              dtype=tf.float32):
  """将整数 token ID 查找并嵌入为稠密向量（词嵌入）。
  
  这是 Transformer 的第一个操作：将每个 token（整数 ID）
  转换为固定维度的稠密向量（embedding）。
  
  实现细节：
  1. 创建 embedding 矩阵（vocab_size × dense_size）
  2. 将梯度转为稠密 Tensor（优化参数服务器通信）
  3. 可选地对 token 进行 dropout（随机将某些 token 置零）
  4. 通过 gather 查找 embedding
  5. 可选地乘以缩放因子（Transformer 用 sqrt(hidden_size) 缩放）
  
  Args:
    x: 整数类型的输入张量，存储 token ID，形状 [batch, length]
    vocab_size: 词表大小（embedding 矩阵的行数）
    dense_size: 嵌入向量维度（d_model，embedding 矩阵的列数）
    name: 变量作用域名称
    reuse: 是否重用已有变量（参数共享时使用）
    multiplier: embedding 向量的缩放系数（Transformer 中为 sqrt(d_model)）
    symbol_dropout_rate: token 级 dropout 概率（在 embedding 前将部分 token 置零）
    embedding_var: 可选的预定义 embedding 矩阵变量
    dtype: 输出数据类型
    
  Returns:
    稠密 embedding 向量，形状 [batch, length, dense_size]
  """
  with tf.variable_scope(
      name, default_name="embedding", values=[x], reuse=reuse, dtype=dtype):
    if embedding_var is None:
      # 创建 embedding 矩阵：[vocab_size, dense_size]
      embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
    # 在反向传播中，将梯度从稀疏 IndexedSlices 转为稠密 Tensor，
    # 以减少参数服务器上的计算量
    if not tf.executing_eagerly():
      embedding_var = convert_gradient_to_tensor(embedding_var)
    # 对 token 进行 dropout（在 embedding 查找之前随机置零某些 token）
    x = dropout_no_scaling(x, 1.0 - symbol_dropout_rate)
    # 通过 gather 查找每个 token 对应的 embedding 向量
    emb_x = gather(embedding_var, x, dtype)
    if multiplier != 1.0:
      # 乘以缩放因子（Transformer 标准：乘以 sqrt(d_model) 来与位置编码量级匹配）
      emb_x *= multiplier
    static_shape = emb_x.shape.as_list()
    if len(static_shape) < 5:
      return emb_x
    assert len(static_shape) == 5
    # 如果有额外的 channel 维度（通常为 1），将其压缩掉
    return tf.squeeze(emb_x, 3)


def shift_right(x, pad_value=None):
  """将 4D 张量在序列维度（第二维）上向右移动一位（Teacher Forcing 的核心操作）。
  
  这是 Transformer 解码器训练时的关键操作，用于实现 Teacher Forcing：
  在训练时，解码器的输入是真实目标序列向右移一位的结果，
  相当于在每个位置上，模型看到的是"前一个真实 token"，然后预测"当前 token"。
  
  例如，目标序列为 [A, B, C, D]，向右移位后变为 [PAD, A, B, C]。
  解码器在位置 0 看到 PAD，预测 A；在位置 1 看到 A，预测 B；以此类推。
  
  注意：这种训练方式与推理时不同。推理时，解码器使用自己生成的上一个 token
  作为下一步的输入（而不是真实 token），这种训练推理不一致称为 "Exposure Bias"。
  
  Args:
    x: 4D 张量，形状 [batch, length, height, depth]
    pad_value: 填充值张量（填在左端），None 时用 0 填充
    
  Returns:
    向右移位后的张量，形状与 x 相同
  """
  if pad_value is None:
    # 在左侧填充一个零（左侧加一行），然后截掉最后一个位置
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
  else:
    # 在左侧拼接 pad_value，然后截掉最后一个位置
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
  return shifted_targets


def shift_right_3d(x, pad_value=None):
  """将 3D 张量在序列维度（第二维）上向右移动一位。
  
  与 shift_right 相同，但适用于 3D 张量 [batch, length, depth]。
  这是 Transformer 解码器的标准输入预处理操作。
  
  Args:
    x: 3D 张量，形状 [batch, length, depth]
    pad_value: 填充值张量，None 时用 0 填充
    
  Returns:
    向右移位后的 3D 张量
  """
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets


def shift_right_2d(x, pad_value=None):
  """将 2D 张量在序列维度（第二维）上向右移动一位。
  
  与 shift_right 相同，但适用于 2D 张量 [batch, length]（如 token ID 序列）。
  
  Args:
    x: 2D 张量，形状 [batch, length]
    pad_value: 填充值张量，None 时用 0 填充
    
  Returns:
    向右移位后的 2D 张量
  """
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0]])[:, :-1]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1]
  return shifted_targets


def conv_stride2_multistep(x, nbr_steps, output_filters, name=None, reuse=None):
  """使用步幅卷积将 x 进行 nbr_steps 次下采样，每次缩小 2 倍。

  使用步幅（stride=2）和卷积核大小为 2 的卷积，以避免反卷积的棋盘格（checkerboard）问题。
  详见：http://distill.pub/2016/deconv-checkerboard/
  
  棋盘格问题：使用转置卷积（deconv）上采样时，由于卷积核的不均匀重叠，
  生成的图像中会出现棋盘格状的伪影。用步幅卷积做下采样可以避免这个问题。

  Args:
    x: 输入张量，形状 [batch, spatial, depth] 或
     [batch, spatial_1, spatial_2, depth]
    nbr_steps: 下采样次数，最终空间维度缩小 2**nbr_steps 倍
    output_filters: 卷积输出通道数
    name: 变量作用域名称
    reuse: 是否重用变量

  Returns:
    (最终特征图, 所有中间特征图的列表)
    最终特征图形状：[batch, spatial / (2**nbr_steps), output_filters] 或
     [batch, spatial_1 / (2**nbr_steps), spatial_2 / (2**nbr_steps), output_filters]
  """
  with tf.variable_scope(
      name, default_name="conv_stride2_multistep", values=[x], reuse=reuse):
    if nbr_steps == 0:
      out = conv(x, output_filters, (1, 1))
      return out, [out]
    hidden_layers = [x]
    for i in range(nbr_steps):
      hidden_layers.append(
          conv(
              hidden_layers[-1],
              output_filters, (2, 2),
              strides=2,
              activation=tf.nn.relu,
              name="conv" + str(i)))
    return hidden_layers[-1], hidden_layers


def deconv_stride2_multistep(x,
                             nbr_steps,
                             output_filters,
                             name=None,
                             reuse=None):
  """使用卷积+reshape 的方式将 x 上采样 2**nbr_steps 倍（避免棋盘格问题）。

  不使用转置卷积（deconv/conv2d_transpose），而是通过：
  - 1D 情况：先用 1x1 卷积扩展通道数到 2 倍，然后 reshape 成 2 倍长度
  - 2D 情况：先用 1x1 卷积扩展通道数到 4 倍，然后用 depth_to_space 进行空间扩展
  
  这种方式避免了转置卷积的棋盘格伪影问题。

  Args:
    x: 输入张量，形状 [batch, spatial, depth] 或
     [batch, spatial_1, spatial_2, depth]
    nbr_steps: 上采样次数，最终空间维度扩大 2**nbr_steps 倍
    output_filters: 输出通道数
    name: 变量作用域名称
    reuse: 是否重用变量

  Returns:
    上采样后的张量，形状 [batch, spatial * (2**nbr_steps), output_filters] 或
     [batch, spatial_1 * (2**nbr_steps), spatial_2 * (2**nbr_steps), output_filters]
  """
  with tf.variable_scope(
      name, default_name="deconv_stride2_multistep", values=[x], reuse=reuse):

    def deconv1d(cur, i):
      cur_shape = shape_list(cur)
      thicker = conv(
          cur,
          output_filters * 2, (1, 1),
          padding="SAME",
          activation=tf.nn.relu,
          name="deconv1d" + str(i))
      return tf.reshape(thicker,
                        [cur_shape[0], cur_shape[1] * 2, 1, output_filters])

    def deconv2d(cur, i):
      thicker = conv(
          cur,
          output_filters * 4, (1, 1),
          padding="SAME",
          activation=tf.nn.relu,
          name="deconv2d" + str(i))
      return tf.depth_to_space(thicker, 2)

    cur = x
    for i in range(nbr_steps):
      if cur.get_shape()[2] == 1:
        cur = deconv1d(cur, i)
      else:
        cur_dim = shape_list(cur)[2]
        if isinstance(cur_dim, int):
          if cur_dim == 1:
            cur = deconv1d(cur, i)
          else:
            cur = deconv2d(cur, i)
        else:
          cur = tf.cond(
              tf.equal(cur_dim, 1),
              lambda idx=i: deconv1d(cur, idx),
              lambda idx=i: deconv2d(cur, idx))
    return cur


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """根据输入形状自动选择 1D 或 2D 卷积的内部实现函数。
  
  这是一个智能卷积包装器，主要功能：
  1. 验证输入必须是 4D 张量（batch, height, width, channels）
  2. 支持 LEFT padding：在序列左侧填充，适用于因果卷积（causal convolution）
     - LEFT padding 保证每个位置只能看到它左边的信息（用于语言模型等自回归任务）
  3. 根据 width=1（1D 序列）还是 width>1（2D 图像）自动选择合适的卷积核
  
  Args:
    conv_fn: 实际的卷积函数（如 tf.layers.conv2d）
    inputs: 4D 输入张量 [batch, height, width, channels]
    filters: 卷积输出通道数
    kernel_size: 卷积核大小（height, width）元组
    **kwargs: 其他传递给 conv_fn 的参数（如 padding、strides、dilation_rate）
    
  Returns:
    卷积输出张量
  """
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4. "
                     "Shape: " + str(static_shape))
  # Add support for left padding.
  if kwargs.get("padding") == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    # Set middle two dimensions to None to prevent convolution from complaining
    inputs.set_shape([static_shape[0], None, None, static_shape[3]])
    kwargs["padding"] = "VALID"

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    name = "{}_{}".format(kwargs.get("name", "conv"), name_suffix)
    original_name = kwargs.pop("name", None)
    original_force2d = kwargs.pop("force2d", None)
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, dilation_rate=(1, 1), **kwargs):
  """标准 2D 卷积（内部使用 Keras Conv2D，支持 LEFT padding 和单/双维自动选择）。
  
  对于 1D 序列输入（width=1），自动使用 (kernel_size[0], 1) 的卷积核。
  支持膨胀卷积（dilation，空洞卷积），可以在不增加参数的情况下扩大感受野。
  
  Args:
    inputs: 4D 输入张量 [batch, height, width, channels]
    filters: 输出通道数
    kernel_size: 卷积核大小（height, width）元组
    dilation_rate: 膨胀率，(height_dilation, width_dilation)
    **kwargs: 其他传递给 Conv2D 的参数（padding、activation、name等）
    
  Returns:
    卷积输出张量
  """
  def _conv2d(x, *args, **kwargs):
    return layers().Conv2D(*args, **kwargs)(x)
  return conv_internal(
      _conv2d,
      inputs,
      filters,
      kernel_size,
      dilation_rate=dilation_rate,
      **kwargs)


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
  """一维卷积：将 3D 张量 [batch, length, depth] 进行 1D 卷积。
  
  内部通过将输入扩展为 4D [batch, length, 1, depth]，
  然后使用 (kernel_size, 1) 的 2D 卷积，再压缩回 3D。
  
  Args:
    inputs: 3D 输入张量 [batch, length, depth]
    filters: 输出通道数
    kernel_size: 卷积核大小（整数）
    dilation_rate: 膨胀率（整数）
    **kwargs: 其他传递给 conv 的参数
    
  Returns:
    3D 卷积输出张量 [batch, length, filters]
  """
  return tf.squeeze(
      conv(tf.expand_dims(inputs, 2), filters, (kernel_size, 1),
           dilation_rate=(dilation_rate, 1), **kwargs),
      2)


def separable_conv(inputs, filters, kernel_size, **kwargs):
  """深度可分离卷积（Depthwise Separable Convolution）。
  
  深度可分离卷积 = 深度卷积（Depthwise Convolution）+ 逐点卷积（Pointwise Convolution）：
  - 深度卷积：对每个通道独立应用空间卷积，计算量更小
  - 逐点卷积：用 1x1 卷积进行通道间的合并
  
  相比标准卷积，可分离卷积有更小的参数量和计算量，
  在 MobileNet、EfficientNet 等轻量级模型中常用。
  
  Args:
    inputs: 4D 输入张量 [batch, height, width, channels]
    filters: 输出通道数
    kernel_size: 卷积核大小
    **kwargs: 其他参数
    
  Returns:
    卷积输出张量
  """
  def _sep_conv2d(x, *args, **kwargs):
    return layers().SeparableConv2D(*args, **kwargs)(x)
  return conv_internal(_sep_conv2d, inputs, filters, kernel_size, **kwargs)


def subseparable_conv(inputs, filters, kernel_size, **kwargs):
  """子可分离卷积：将输入分成多个块，分别应用卷积后合并。
  
  通过 separability 参数控制分块数量：
  - separability=0 或 None：等价于 separable_conv（完全可分离卷积）
  - separability>0：将输入分成 abs(separability) 块，
    每块用普通 2D 卷积处理，最后用 1x1 卷积合并
  - separability<0：将输入分成 abs(separability) 块，
    每块用可分离卷积处理，然后拼接
  
  这种分块方式在保持输出大小不变的情况下减少计算量。
  
  Args:
    inputs: 4D 输入张量
    filters: 输出通道数
    kernel_size: 卷积核大小
    **kwargs: 其他参数，可包含 separability 参数
    
  Returns:
    卷积输出张量
  """

  def conv_fn(inputs, filters, kernel_size, **kwargs):
    """Sub-separable convolution, splits into separability-many blocks."""
    separability = None
    if "separability" in kwargs:
      separability = kwargs.pop("separability")
    if separability:
      parts = []
      abs_sep = separability if separability > 0 else -1 * separability
      for split_idx, split in enumerate(tf.split(inputs, abs_sep, axis=3)):
        with tf.variable_scope("part_%d" % split_idx):
          if separability > 0:
            parts.append(
                layers().Conv2D(filters // separability, kernel_size,
                                **kwargs)(split))
          else:
            parts.append(
                layers().SeparableConv2D(filters // abs_sep,
                                         kernel_size, **kwargs)(split))
      if separability > 1:
        result = layers().Conv2D(filters, (1, 1))(tf.concat(parts, axis=3))
      elif abs_sep == 1:  # If we have just one block, return it.
        assert len(parts) == 1
        result = parts[0]
      else:
        result = tf.concat(parts, axis=3)
    else:
      result = layers().SeparableConv2D(filters, kernel_size,
                                        **kwargs)(inputs)
    if separability is not None:
      kwargs["separability"] = separability
    return result

  return conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs)


def tpu_conv1d(inputs, filters, kernel_size, padding="SAME", name="tpu_conv1d"):
  """适用于 TPU 的 1D 卷积实现（将卷积展开为多个密集的矩阵乘法）。
  
  标准 conv1d 将大卷积核实现为多个 kernel_size 次矩阵乘法的加和：
  对于 kernel_size=k，分别对输入的不同偏移位置应用全连接，
  然后将 k 个结果相加并缩放，这在 TPU 上比直接卷积彦更高效。
  
  实现原理：
  - conv1d(内核大小=k) = sum_{i=0}^{k-1} dense(输入左移 i 位)
  - 相当于横向展开卷积核
  
  Args:
    inputs: 3D 输入张量 [batch, length, input_depth]
    filters: 输出特征维度
    kernel_size: 卷积核大小（整数）
    padding: 填充方式，"SAME"（两侧填充）或 "LEFT"（左侧填充，因果卷积）
    name: 变量作用域名称
    
  Returns:
    3D 张量 [batch, length, filters]
  """
  if kernel_size == 1:
    return dense(inputs, filters, name=name, use_bias=True)
  if padding == "SAME":
    assert kernel_size % 2 == 1
    first_offset = -((kernel_size - 1) // 2)
  else:
    assert padding == "LEFT"
    first_offset = -(kernel_size - 1)
  last_offset = first_offset + kernel_size - 1
  results = []
  padded = tf.pad(inputs, [[0, 0], [-first_offset, last_offset], [0, 0]])
  for i in range(kernel_size):
    shifted = tf.slice(padded, [0, i, 0], tf.shape(inputs)) if i else inputs
    shifted.set_shape(inputs.get_shape())
    results.append(
        dense(shifted, filters, use_bias=(i == 0), name=name + "_%d" % i))
  ret = tf.add_n(results)
  ret *= kernel_size**-0.5
  return ret


def layer_norm_vars(filters):
  """创建层归一化（Layer Normalization）所需的可训练变量。
  
  层归一化有两个可训练参数：
  - scale（缩放）：初始值为 1，在归一化后对每个特征进行缩放
  - bias（偏置）：初始值为 0，在缩放后加上偏置
  
  Args:
    filters: 特征维度大小（归一化的最后一维的大小）
    
  Returns:
    (scale, bias): 缩放参数和偏置参数，形状均为 [filters]
  """
  # scale 初始化为 1（归一化后不改变量级）
  scale = tf.get_variable(
      "layer_norm_scale", [filters], initializer=tf.ones_initializer())
  # bias 初始化为 0（归一化后不偏移）
  bias = tf.get_variable(
      "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias


def layer_norm_compute(x, epsilon, scale, bias, layer_collection=None):
  """层归一化的核心数学计算。
  
  层归一化（Layer Normalization）是 Transformer 中至关重要的操作，
  它对每个样本的特征维度进行归一化（而不是像 BatchNorm 那样对 batch 维度归一化）。
  
  数学公式：
    mean = mean(x, axis=-1)        # 计算特征维度的均值
    var = var(x, axis=-1)          # 计算特征维度的方差
    norm_x = (x - mean) / sqrt(var + epsilon)  # 归一化
    output = norm_x * scale + bias  # 可学习的缩放和偏移
  
  层归一化的优点：
  - 不依赖 batch 大小（批归一化依赖 batch，推理时不稳定）
  - 适合序列模型（RNN、Transformer），每个时间步独立归一化
  - 帮助训练更稳定、加速收敛
  
  Args:
    x: 输入张量，最后一维是特征维度
    epsilon: 数值稳定性参数，防止除以零（通常为 1e-6）
    scale: 可学习的缩放参数，形状 [filters]
    bias: 可学习的偏置参数，形状 [filters]
    layer_collection: 可选，Kronecker 因式分解近似曲率（KFAC）的层集合
    
  Returns:
    归一化后的输出张量，形状与 x 相同
  """
  # 保存参数（在下面的类型转换之前保存，供 KFAC 使用）
  params = (scale, bias)

  # 将 epsilon、scale、bias 转换为与 x 相同的数据类型（如 bfloat16）
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  # 计算特征维度的均值（保持维度以便广播）
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  # 计算特征维度的方差（使用 squared_difference 避免减法精度问题）
  variance = tf.reduce_mean(
      tf.squared_difference(x, mean), axis=[-1], keepdims=True)
  # 归一化：减均值，除以标准差（rsqrt = 1/sqrt，加 epsilon 防止除零）
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

  # 应用可学习的缩放（scale）和偏移（bias）
  output = norm_x * scale + bias

  return output


def layer_norm(x,
               filters=None,
               epsilon=1e-6,
               name=None,
               reuse=None,
               layer_collection=None):
  """对张量 x 在最后一个维度上进行层归一化。
  
  这是 Transformer 中最常用的归一化方式，在每层的输入或输出上应用。
  层归一化可以：
  - 缓解梯度消失/爆炸问题
  - 减少对初始化的敏感性
  - 加速训练收敛
  
  Args:
    x: 输入张量（任意维度，但最后一维是特征维度）
    filters: 特征维度大小，None 时自动推断
    epsilon: 归一化的数值稳定参数（1e-6）
    name: 变量作用域名称
    reuse: 是否重用已有变量
    layer_collection: 可选，KFAC 的层集合
    
  Returns:
    归一化后的张量，形状与 x 相同
  """
  if filters is None:
    filters = shape_list(x)[-1]  # 自动推断最后一维的大小
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    # 创建归一化参数（scale 和 bias）
    scale, bias = layer_norm_vars(filters)
    # 执行层归一化计算
    return layer_norm_compute(x, epsilon, scale, bias,
                              layer_collection=layer_collection)


def group_norm(x, filters=None, num_groups=8, epsilon=1e-5):
  """组归一化（Group Normalization），论文：https://arxiv.org/abs/1803.08494。
  
  组归一化是层归一化和实例归一化的折中：
  - 将特征通道分为若干组，在每组内进行归一化
  - 当 num_groups=1 时，等价于层归一化（在整个特征维度归一化）
  - 当 num_groups=filters 时，等价于实例归一化（每个通道单独归一化）
  
  优点：
  - 不依赖 batch 大小（优于批归一化）
  - 对图像任务比层归一化更有效（考虑了空间信息）
  
  Args:
    x: 4D 输入张量，形状 [batch, height, width, filters]
    filters: 通道数，None 时自动推断
    num_groups: 分组数量（filters 必须能被 num_groups 整除）
    epsilon: 数值稳定性参数
    
  Returns:
    归一化后的 4D 张量，形状与 x 相同
  """
  x_shape = shape_list(x)
  if filters is None:
    filters = x_shape[-1]
  assert len(x_shape) == 4  # 要求 4D 输入
  assert filters % num_groups == 0  # 通道数必须能被分组数整除
  # 创建可学习的缩放和偏置参数
  scale = tf.get_variable(
      "group_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "group_norm_bias", [filters], initializer=tf.zeros_initializer())
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  # 将通道维度分组：[batch, H, W, filters] -> [batch, H, W, groups, filters/groups]
  x = tf.reshape(x, x_shape[:-1] + [num_groups, filters // num_groups])
  # 在 H、W 和组内通道维度上计算均值和方差（不跨组归一化）
  mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  # 将形状恢复并应用缩放偏置
  return tf.reshape(norm_x, x_shape) * scale + bias


def noam_norm(x, epsilon=1.0, name=None):
  """Noam 归一化：基于 L2 归一化的简化版层归一化，无可学习参数。
  
  这是一种简单的归一化方式：
  - 使用 L2 归一化（除以 L2 范数）代替标准差归一化
  - 乘以 sqrt(d) 来维持原始量级
  - 没有可学习的 scale 和 bias 参数（比标准层归一化参数更少）
  
  Args:
    x: 输入张量
    epsilon: 防止除零的小常数
    name: 命名域名称
    
  Returns:
    归一化后的张量，形状与 x 相同
  """
  with tf.name_scope(name, default_name="noam_norm", values=[x]):
    shape = x.get_shape()
    ndims = len(shape)
    # L2 归一化后乘以 sqrt(d) 恢复量级
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
        to_float(shape[-1])))


def l2_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """基于 L2 范数的层归一化，带可学习的 scale 和 bias 参数。
  
  与标准层归一化类似，但使用 L2 范数（平方和的平方根）武代标准差来归一化。
  
  Args:
    x: 输入张量
    filters: 特征维度，None 时自动推断
    epsilon: 防止除零的小常数
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    归一化后的张量，形状与 x 相同
  """
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(name, default_name="l2_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "l2_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "l2_norm_bias", [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    l2norm = tf.reduce_sum(
        tf.squared_difference(x, mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(l2norm + epsilon)
    return norm_x * scale + bias


def apply_spectral_norm(x):
  """使用谱范数归一化 x（常用于 GAN 的判别器中）。

  实现参考论文 https://arxiv.org/abs/1802.05957 的算法 1。
  
  谱归一化（Spectral Normalization）是一种权重归一化方法，
  它将权重除以权重矩阵的最大奇异值（谱范数 = 最大奇异值），
  使网络满足 Lipschitz 连续条件，从而稳定 GAN 训练。
  
  算法使用冪迭q方法进行奇异分解，通过维护一个向量 u 的近似不断更新。
  
  如果 x 不是 2D 张量，就将其 reshape 使通道数（最后一维）不变。

  Args:
    x: 最后一维等于通道数的张量（如卷积权重）

  Returns:
    x: 形状与 x 相同的谱范数归一化后的张量
    assign_op: 每步训练后需要运行的于更新向量 "u" 的操作
  """
  weights_shape = shape_list(x)
  other, num_filters = tf.reduce_prod(weights_shape[:-1]), weights_shape[-1]

  # Reshape into a 2-D matrix with outer size num_filters.
  weights_2d = tf.reshape(x, (other, num_filters))

  # v = Wu / ||W u||
  with tf.variable_scope("u", reuse=tf.AUTO_REUSE):
    u = tf.get_variable(
        "u", [num_filters, 1],
        initializer=tf.truncated_normal_initializer(),
        trainable=False)
  v = tf.nn.l2_normalize(tf.matmul(weights_2d, u))

  # u_new = vW / ||v W||
  u_new = tf.nn.l2_normalize(tf.matmul(tf.transpose(v), weights_2d))

  # s = v*W*u
  spectral_norm = tf.squeeze(
      tf.matmul(tf.transpose(v), tf.matmul(weights_2d, tf.transpose(u_new))))

  # set u equal to u_new in the next iteration.
  assign_op = tf.assign(u, tf.transpose(u_new))
  return tf.divide(x, spectral_norm), assign_op


def apply_norm(x, norm_type, depth, epsilon, layer_collection=None):
  """根据 norm_type 参数选择并应用对应的归一化方法。
  
  这是一个归一化方法的统一分发函数，根据 norm_type 选择：
  - "layer": 层归一化（Layer Norm，Transformer 默认方式）
  - "group": 组归一化（Group Norm，适合图像任务）
  - "batch": 批归一化（Batch Norm，适合 CNN）
  - "noam":  Noam 归一化（简化版层归一化，无可学习参数）
  - "l2":    L2 归一化变体
  - "none":  不归一化，直接返回输入
  
  Args:
    x: 输入张量
    norm_type: 归一化类型字符串（见上方描述）
    depth: 特征维度大小（层归一化使用）
    epsilon: 归一化的数值稳定参数
    layer_collection: 可选，KFAC 优化器的层集合（仅层归一化支持）
    
  Returns:
    归一化后的张量，形状与 x 相同
    
  Raises:
    ValueError: 如果 norm_type 不在支持的类型中
  """
  if layer_collection is not None:
    assert norm_type == "layer"  # KFAC 目前只支持层归一化
  if norm_type == "layer":
    # 层归一化：Transformer 的标准选择
    return layer_norm(
        x, filters=depth, epsilon=epsilon, layer_collection=layer_collection)
  if norm_type == "group":
    # 组归一化：适合图像任务
    return group_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "batch":
    # 批归一化：适合 CNN，不依赖序列长度
    return layers().BatchNormalization(epsilon=epsilon)(x)
  if norm_type == "noam":
    # Noam 归一化：简化版，无可学习参数
    return noam_norm(x, epsilon)
  if norm_type == "l2":
    # L2 归一化
    return l2_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "none":
    return x  # 不归一化，直接返回
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'lr', 'none'.")


def zero_add(previous_value, x, name=None, reuse=None):
  """零初始化的跳跃连接（Zero-Init Residual Connection）。

  返回 previous_value + gamma * x，其中 gamma 是可训练标量且初始化为 0。
  
  用途：当将新模块插入到已训练模型中时，希望新模块初始时
  不改变原模型的行为。gamma=0 保证初始时输出等于 previous_value，
  随着训练 gamma 逐渐增大，新模块的贡献满渐增强。
  
  这是一种更安全的迎来步进加模块的方式。

  Args:
    previous_value:  A tensor.
    x: A tensor.
    name: name of variable scope; defaults to zero_add.
    reuse: reuse scope.

  Returns:
    previous_value + gamma * x.
  """
  with tf.variable_scope(name, default_name="zero_add", reuse=reuse):
    gamma = tf.get_variable("gamma", (), initializer=tf.zeros_initializer())
    return previous_value + gamma * x


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None,
                         layer_collection=None):
  """对层的输入或输出依次应用一系列变换操作（层预处理/后处理的核心函数）。
  
  这是 Transformer 中每一层开始和结束时的关键操作序列。
  变换序列由字符串指定，每个字符代表一种操作：
    'a': 添加 previous_value（残差连接 Residual Connection）
         将当前层的输入加到输出上，这是解决梯度消失问题的核心技术
    'n': 应用归一化（Normalization，类型由 norm_type 决定，通常是层归一化）
    'd': 应用 Dropout（随机丢弃神经元，用于正则化，防止过拟合）
    'z': 零初始化残差连接（Zero Add，初始化时不改变原始模型行为）
  
  Transformer 的两种常见配置：
  1. Post-LayerNorm（原始论文）: preprocess="none", postprocess="dan"
     流程: x -> sublayer -> dropout -> add(x) -> layernorm
     即：输出 = layernorm(x + dropout(sublayer(x)))
  
  2. Pre-LayerNorm（现代改进）: preprocess="n", postprocess="da"
     流程: layernorm(x) -> sublayer -> dropout -> add(x)
     即：输出 = x + dropout(sublayer(layernorm(x)))
     这种方式训练更稳定（梯度流动更好）。
  
  例如，如果 sequence=="dna"，则操作为：
    previous_value + normalize(dropout(x))
    即：先对 x 做 dropout，再归一化，最后加上残差连接
  
  Args:
    previous_value: 残差连接中被加的张量（通常是层输入）
    x: 需要被变换的张量（通常是层输出）
    sequence: 操作序列字符串（如 "dan"、"n"、"da"）
    dropout_rate: Dropout 率（如 0.1 表示 10% 的神经元被丢弃）
    norm_type: 归一化类型（见 apply_norm()）
    depth: x 最后一维的大小（用于归一化）
    epsilon: 归一化的数值稳定参数
    default_name: 默认变量作用域名称
    name: 变量作用域名称（None 时使用 default_name）
    dropout_broadcast_dims: 可选的整数列表（维度数 < 3），
                           指定在哪些维度上广播 dropout 决策（节省内存）
    layer_collection: tensorflow_kfac 的 LayerCollection，仅 KFAC 优化器使用
    
  Returns:
    经过序列变换后的张量
  """
  with tf.variable_scope(name, default_name=default_name):
    if sequence == "none":
      return x  # "none" 表示不做任何处理
    # 依次执行序列中的每个操作
    for c in sequence:
      if c == "a":
        # 残差连接：将层输入加到当前张量上
        # 这是 ResNet 思想：y = x + F(x)，让模型学习残差而非完整映射
        x += previous_value
      elif c == "z":
        # 零初始化残差连接：output = previous_value + gamma * x
        # gamma 初始化为 0，确保初始时模块不改变原始输出
        # 适合将新模块插入已训练模型中（不破坏原有性能）
        x = zero_add(previous_value, x)
      elif c == "n":
        # 归一化：对张量进行归一化（通常是层归一化）
        x = apply_norm(
            x, norm_type, depth, epsilon, layer_collection=layer_collection)
      else:
        assert c == "d", ("Unknown sequence step %s" % c)
        # Dropout：以 dropout_rate 的概率随机丢弃神经元
        x = dropout_with_broadcast_dims(
            x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    return x


def layer_preprocess(layer_input, hparams, layer_collection=None):
  """Apply layer preprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_preprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    hparams: a hyperparameters object.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.

  Returns:
    a Tensor
  """
  assert "a" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  assert "z" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence=hparams.layer_preprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_type=hparams.norm_type,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_prepostprocess",
      layer_collection=layer_collection)


def layer_postprocess(layer_input, layer_output, hparams):
  """Apply layer postprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_postprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.

  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=hparams.layer_postprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_type=hparams.norm_type,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_postprocess")


def conv_block_internal(conv_fn,
                        inputs,
                        filters,
                        dilation_rates_and_kernel_sizes,
                        first_relu=True,
                        use_elu=False,
                        separabilities=None,
                        **kwargs):
  """一个卷积块：多层卷积的封装，支持膨胀卷积、层归一化和各种激活函数。

  这是多种卷积块（conv_block、conv1d_block 等）的通用内部实现。
  每个卷积层的操作顺序：激活函数 -> 掩码 -> 卷积 -> 层归一化
  
  Args:
    conv_fn: 卷积函数，如 conv 或 separable_conv
    inputs: 输入张量
    filters: 卷积输出通道数
    dilation_rates_and_kernel_sizes: 展张率和卷积核大小的列表，格式为 [(dilation, (k_h, k_w)), ...]
      列表中的每个元素对应一层卷积
    first_relu: 是否在第一层卷积前应用激活函数（默认为 True）
    use_elu: 是否使用 ELU 代替 ReLU（默认为 False）
    separabilities: 可分离度列表（每层一个），用于 subseparable_conv
    **kwargs: 其他传递给卷积函数的参数（如 pooling、padding 等）

  Returns:
     卷积块的输出张量
  """

  name = kwargs.pop("name") if "name" in kwargs else None
  mask = kwargs.pop("mask") if "mask" in kwargs else None

  # Usage for normalize_fn kwarg:
  # if not specified, use layer norm
  # if given normalize_fn=None, don't use any normalization
  # if given normalize_fn=norm, use the specified norm function

  use_layer_norm = "normalizer_fn" not in kwargs
  norm = kwargs.pop("normalizer_fn", None)
  use_normalizer_fn = use_layer_norm or norm

  if use_layer_norm:
    norm = lambda x, name: layer_norm(x, filters, name=name)

  with tf.variable_scope(name, "conv_block", [inputs]):
    cur, counter = inputs, -1
    for dilation_rate, kernel_size in dilation_rates_and_kernel_sizes:
      counter += 1
      if first_relu or counter > 0:
        cur = tf.nn.elu(cur) if use_elu else tf.nn.relu(cur)
      if mask is not None:
        cur *= mask
      if separabilities:
        cur = conv_fn(
            cur,
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            name="conv_block_%d" % counter,
            use_bias=norm is None,
            separability=separabilities[counter],
            **kwargs)
      else:
        cur = conv_fn(
            cur,
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            name="conv_block_%d" % counter,
            use_bias=norm is None,
            **kwargs)
      if use_normalizer_fn:
        cur = norm(cur, name="conv_block_norm_%d" % counter)
    return cur


def conv_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """标准 2D 卷积块：使用标准 2D 卷积的多层卷积组合。
  
  Args:
    inputs: 输入张量
    filters: 输出通道数
    dilation_rates_and_kernel_sizes: [(dilation, kernel_size), ...] 列表
    **kwargs: 其他参数
  """
  return conv_block_internal(conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def conv1d_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """标准 1D 卷积块：使用 1D 卷积的多层卷积组合（适合序列模型）。
  
  Args:
    inputs: 输入张量
    filters: 输出通道数
    dilation_rates_and_kernel_sizes: [(dilation, kernel_size), ...] 列表
    **kwargs: 其他参数
  """
  return conv_block_internal(conv1d, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def separable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                         **kwargs):
  """深度可分离卷积块：使用可分离卷积的多层卷积组合，比标准卷积更高效。
  
  Args:
    inputs: 输入张量
    filters: 输出通道数
    dilation_rates_and_kernel_sizes: [(dilation, kernel_size), ...] 列表
    **kwargs: 其他参数
  """
  return conv_block_internal(separable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def subseparable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                            **kwargs):
  """子可分离卷积块：使用子可分离卷积的多层卷积组合。
  
  Args:
    inputs: 输入张量
    filters: 输出通道数
    dilation_rates_and_kernel_sizes: [(dilation, kernel_size), ...] 列表
    **kwargs: 其他参数
  """
  return conv_block_internal(subseparable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def pool(inputs, window_size, pooling_type, padding, strides=(1, 1)):
  """池化操作，支持 LEFT padding（序列左侧填充，用于因果模型）。
  
  支持的池化类型：最大池化（MAX）、平均池化（AVG）等。
  LEFT padding 保证当前位置的池化只能看到它左边的信息，符合自回归的因果性。
  
  Args:
    inputs: 4D 输入张量 [batch, height, width, channels]
    window_size: 池化窗口大小 (height, width)
    pooling_type: 池化类型字符串（"MAX" 或 "AVG"）
    padding: 填充方式，"SAME"、"VALID" 或 "LEFT"
    strides: 步幅 (height_stride, width_stride)
    
  Returns:
    池化后的张量
  """
  with tf.name_scope("pool", values=[inputs]):
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
      raise ValueError("Inputs to conv must have statically known rank 4.")
    # Add support for left padding.
    if padding == "LEFT":
      assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1
      if len(static_shape) == 3:
        width_padding = 2 * (window_size[1] // 2)
        padding_ = [[0, 0], [width_padding, 0], [0, 0]]
      else:
        height_padding = 2 * (window_size[0] // 2)
        cond_padding = tf.cond(
            tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
            lambda: tf.constant(2 * (window_size[1] // 2)))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding_ = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
      inputs = tf.pad(inputs, padding_)
      inputs.set_shape([static_shape[0], None, None, static_shape[3]])
      padding = "VALID"

  return tf.nn.pool(inputs, window_size, pooling_type, padding, strides=strides)


def conv_block_downsample(x,
                          kernel,
                          strides,
                          padding,
                          separability=0,
                          name=None,
                          reuse=None):
  """实现下采样卷积块，类似 Xception 网络的 Exit Flow 结构。
  
  结构：
  1. 跳跃连接（res）：单层卷积，输出通道数 1.25*hidden_size
  2. 主干道：两层子可分离卷积 + 池化，加上 res
  3. 再两层子可分离卷积，输出通道数终到 2.5*hidden_size
  
  下采样通过 strides 参数实现，池化层进一步缩小空间分辨率。
  
  Args:
    x: 输入张量
    kernel: 卷积核大小
    strides: 步幅（实现下采样）
    padding: 填充方式
    separability: 子可分离度
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    下采样后的张量
  """
  with tf.variable_scope(
      name, default_name="conv_block_downsample", values=[x], reuse=reuse):
    hidden_size = int(x.get_shape()[-1])
    res = conv_block(
        x,
        int(1.25 * hidden_size), [((1, 1), kernel)],
        padding=padding,
        strides=strides,
        name="res_conv")

    x = subseparable_conv_block(
        x,
        hidden_size, [((1, 1), kernel)],
        padding=padding,
        separability=separability,
        name="conv0")
    x = subseparable_conv_block(
        x,
        int(1.25 * hidden_size), [((1, 1), kernel)],
        padding=padding,
        separability=separability,
        name="conv1")
    x = pool(x, kernel, "MAX", padding, strides=strides)

    x += res

    x = subseparable_conv_block(
        x,
        2 * hidden_size, [((1, 1), kernel)],
        first_relu=False,
        padding=padding,
        separability=separability,
        name="conv2")
    x = subseparable_conv_block(
        x,
        int(2.5 * hidden_size), [((1, 1), kernel)],
        padding=padding,
        separability=separability,
        name="conv3")
    return x


def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16):
  """创建不同频率的正弦信号张量（用于生成位置编码）。

  Transformer 的位置编码使用不同频率的正弦和余弦函数的组合表示序列位置。
  对于位置 i 和频率索引 k：
  - PE(i, 2k) = sin(i / timescale_k)
  - PE(i, 2k+1) = cos(i / timescale_k)
  
  其中 timescale 在 [min_timescale, max_timescale] 之间成几何级数展开。
  
  为什么这样设计？
  - 低频分量：区分远距的位置关系
  - 高频分量：区分近距的位置关系
  - 正余弦组合：可以用加法公式表示相对位置，允许模型学习相对位置信息
  
  Args:
    length: 序列长度（位置数）
    min_timescale: 最小时频（最高频率）
    max_timescale: 最大时频（最低频率）
    num_timescales: 时频数量，输出向量维度 = 2 * num_timescales

  Returns:
    形状为 (length, 2*num_timescales) 的位置编码张量
  """
  positions = to_float(tf.range(length))
  log_timescale_increment = (
      math.log(max_timescale / min_timescale) / (num_timescales - 1))
  inv_timescales = min_timescale * tf.exp(
      to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)


def add_timing_signal(x, min_timescale=1, max_timescale=1e4, num_timescales=16):
  """将不同频率的正弦波位置编码加入输入张量中（Transformer 位置编码）。

  这使得注意力能够学习利用绝对和相对位置信息。
  位置信号应该被加到注意力的源和目标序列的前身。

  支持相对位置的原因：sin(x+y) 和 cos(x+y) 可以用 y、sin(x)、cos(x) 表示，
  因此模型可以由两个位置的编码推导它们的相对距离。

  具体实现：使用从 min_timescale 到 max_timescale 的几何级数列时频序列。
  对每个时频，生成两个正弦信号 sin(timestep/timescale) 和 cos(timestep/timescale)。
  所有正弦信号拼接在深度维度，并填充零以匹配输入深度，然后加入输入。

  Args:
    x: 形状为 [?, length, ?, depth] 的张量
    min_timescale: 最小时频（最大频率）
    max_timescale: 最大时频（最小频率）
    num_timescales: 时频数，必须 <= depth/2

  Returns:
    与 x 形状相同的张量，已加入位置编码
  """
  length = shape_list(x)[1]
  depth = shape_list(x)[3]
  signal = get_timing_signal(length, min_timescale, max_timescale,
                             num_timescales)
  padded_signal = tf.pad(signal, [[0, 0], [0, depth - 2 * num_timescales]])
  return x + tf.reshape(padded_signal, [1, length, 1, depth])


def mask_from_embedding(emb):
  """从嵌入向量生成 padding 掩码（笔展位置为 0，其他位置为 1）。

  symbol_modality 被修改为对填充位置返回全零嵌入向量。
  这个函数利用这一特性，通过检测嵌入向量是否全为零来判断填充位置。

  Args:
    emb: 嵌入张量，形状 [batch, width, height, depth]
  Returns:
    0.0/1.0 浮点型掩码，形状 [batch, width, height, 1]
    - 1.0：非填充位置（有效内容）
    - 0.0：填充位置（PAD token）
  """
  return weights_nonzero(tf.reduce_sum(tf.abs(emb), axis=3, keepdims=True))


def length_from_embedding(emb):
  """从嵌入张量计算每个序列的实际长度（不包括 padding）。

  Args:
    emb: 序列嵌入张量，形状 [batch, max_time, 1, depth]
  Returns:
    形状为 [batch] 的整数张量，表示每个序列的实际长度
  """
  return tf.cast(tf.reduce_sum(mask_from_embedding(emb), [1, 2, 3]), tf.int32)


def mask_pos_gt(source_length, target_length):
  """生成模果序列 source_pos > target_pos 时为 1.0，否则为 0.0 的掩码。
  
  这个掩码将 target_pos < source_pos 的位置隐蚀（也就是 target 位置前面的 source 位置）。
  通常用于注意力的序列不对空掌。

  Args:
    source_length: source 序列长度
    target_length: target 序列长度
  Returns:
    形状为 [1, target_length, source_length] 的浮点掩码张量
  """
  return tf.expand_dims(
      tf.cast(tf.greater(tf.expand_dims(tf.range(target_length), axis=0),
                         tf.expand_dims(tf.range(source_length), axis=1)),
              dtype=tf.float32), axis=0)


def mask_leq(target_length, source_length):
  """生成 source_pos <= target_pos 时为 1.0 的下三角掩码（包括对角线）。
  
  这是 Transformer 解码器自回归注意力的核心掩码：
  确保位置 i 的 token 只能看到位置 0‘1 、...i 的 token（不能看未来）。

  Args:
    target_length: target 序列长度（行数）
    source_length: source 序列长度（列数）
  Returns:
    形状为 [1, target_length, source_length] 的下三角浮点掩码张量
  """
  return ones_matrix_band_part(
      target_length,
      source_length,
      -1,
      0,
      out_shape=[1, target_length, source_length])


def mask_pos_lt(source_length, target_length):
  """生成 source_pos < target_pos 时为 1.0，否则为 0.0 的严格下三角掩码（不包括对角线）。
  
  与 mask_leq 类似，但是不包括对角线（source_pos = target_pos 时为 0）。
  用于需要严格因果性的场景（不能看当前位置）。

  Args:
    source_length: source 序列长度
    target_length: target 序列长度
  Returns:
    形状为 [1, target_length, source_length] 的浮点掩码张量
  """
  return tf.expand_dims(
      tf.cast(tf.less(tf.expand_dims(tf.range(target_length), axis=0),
                      tf.expand_dims(tf.range(source_length), axis=1)),
              dtype=tf.float32), axis=0)


def relu_density_logit(x, reduce_dims):
  """logit(density(x))，用于可视化 ReLU 激活密度的直方图。

  计算 ReLU 激活密度的 logit （反转数几何平均），用于特征分布的可视化。
  
  长期处于 0 付近的密度或 1 付近的密度均可能表明梯度消失。
  返回的 logit 值易于在气步图中观察。

  Args:
    x: 通常是 tf.relu 输出的张量
    reduce_dims: 需要除去的维度列表

  Returns:
    张量，内容是 logit(密度)
  """
  frac = tf.reduce_mean(to_float(x > 0.0), reduce_dims)
  scaled = tf.log(frac + math.exp(-10)) - tf.log((1.0 - frac) + math.exp(-10))
  return scaled


def maybe_zero_out_padding(inputs, kernel_size, nonpadding_mask):
  """如果必要，将序列填充位置的卷积输入置零。

  对于 kernel_size != 1 的卷积，填充位置上的张量可能会影响邻近位置（内死区渗出）。
  这个函数将填充位置的输入置零，避免填啂位置影响卷积结果。

  Args:
    inputs: 张量，形状 [batch, length, ...]
    kernel_size: 卷积核大小（整数或整数对）
    nonpadding_mask: 非填啂位置的浮点掩码，形状 [batch, length]（非填啂为 1，填啂为 0）

  Returns:
    与 inputs 形状相同的张量，填啂位置已置零
  """
  if (kernel_size != 1 and kernel_size != (1, 1) and
      nonpadding_mask is not None):
    while nonpadding_mask.get_shape().ndims < inputs.get_shape().ndims:
      nonpadding_mask = tf.expand_dims(nonpadding_mask, -1)
    return inputs * nonpadding_mask

  return inputs


def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     layer_collection=None,
                     name=None):
  """前馈神经网络（FFN）：全连接 -> ReLU 激活 -> Dropout -> 全连接。
  
  这是 Transformer 设计中每个层的前馈网络（Feed-Forward Network, FFN）的实现。
  原始论文中， FFN = Linear(ReLU(Linear(x)))，通常隐藏层维度 filter_size = 4 * d_model。
  
  操作流程：
  1. dense(inputs, filter_size) + ReLU ：将特征维度扩展并激活
  2. dropout：训练时随机丢弃少数神经元
  3. dense(h, output_size)：将特征维度庋减回原始大小
  
  注意：使用 "conv1"、"conv2" 作为变量名称是历史原因，实际上是全连接层。
  
  Args:
    inputs: 输入张量，形状 [batch, length, depth]
    filter_size: FFN 隐藏层大小（通常为 4 * d_model）
    output_size: FFN 输出大小（通常为 d_model）
    output_activation: 输出激活函数，None 表示线性（无激活）
    dropout: Dropout 率，0.0 表示不使用
    dropout_broadcast_dims: Dropout 的广播维度列表
    layer_collection: KFAC 的层集合
    name: 变量作用域名称
    
  Returns:
    FFN 输出张量，形状 [batch, length, output_size]
  """
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  layer_name = "%s_{}" % name if name else "{}"
  h = dense(
      inputs,
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      layer_collection=layer_collection,
      name=layer_name.format("conv1"))

  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      layer_collection=layer_collection,
      name=layer_name.format("conv2"))
  return o


def dense_dropconnect(inputs,
                      output_size,
                      dropconnect_dropout=0.0,
                      name="dense_dropconnect",
                      **kwargs):
  """带 DropConnect 的全连接层。
  
  DropConnect 是 Dropout 的变种：
  - Dropout：训练时随机将一些神经元的输出置零
  - DropConnect：训练时随机将一些权重（连接）置零
  具有类似的正则化效果，但在某些情况下也许有更好效果。
  
  实现上，将 Dropout 作为卷积核的正则化函数应用于权重张量。
  
  Args:
    inputs: 输入张量
    output_size: 输出维度
    dropconnect_dropout: DropConnect 的 dropout 率，0.0 表示不使用
    name: 变量作用域名称
    **kwargs: 其他传递给 dense 的参数
    
  Returns:
    全连接层输出张量
  """

  if dropconnect_dropout != 0.0:
    tf.logging.info("Applying dropconnect as the kernel regularization.")
    kwargs["kernel_regularizer"] = functools.partial(
        tf.nn.dropout, keep_prob=1.0 - dropconnect_dropout)

  return dense(inputs, output_size, use_bias=True, name=name, **kwargs)


def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None,
                   cache=None,
                   decode_loop_step=None):
  """卷积 + ReLU 激活 + 卷积的前馈层（支持缓存加速解码）。

  与 dense_relu_dense 类似，但使用卷积代替全连接，适合序列中具有局部层次结构的任务。
  
  支持连接缓存（Key-Value Cache），用于 Transformer 解码器的快速推理：
  - 常规模式：将当前输入倒接到 cache 中，保留最近 kernel_size 个时间步
  - TPU 模式（decode_loop_step != None）：使用原地更新代替拼接，附合 TPU 内存局限

  Args:
    inputs: 输入张量
    filter_size: FFN 隐藏层大小
    output_size: FFN 输出大小
    first_kernel_size: 第一个卷积的卷积核大小
    second_kernel_size: 第二个卷积的卷积核大小
    padding: 填充方式
    nonpadding_mask: 非填啂位置的浮点掩码
    dropout: Dropout 率
    name: 变量作用域名称
    cache: 包含之前注意力结果的字典，用于快速解码
    decode_loop_step: 解码循环的步数，仅在 TPU 推理时使用

  Returns:
    输出张量
  """
  with tf.variable_scope(name, "conv_relu_conv", [inputs]):
    inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)

    if cache:
      if decode_loop_step is None:
        inputs = cache["f"] = tf.concat([cache["f"], inputs], axis=1)
      else:
        # Inplace update is required for inference on TPU.
        # Inplace_ops only supports inplace_update on the first dimension.
        # The performance of current implementation is better than updating
        # the tensor by adding the result of matmul(one_hot,
        # update_in_current_step)
        tmp_f = tf.transpose(cache["f"], perm=[1, 0, 2])
        tmp_f = inplace_ops.alias_inplace_update(
            tmp_f,
            decode_loop_step * tf.shape(inputs)[1],
            tf.transpose(inputs, perm=[1, 0, 2]))
        inputs = cache["f"] = tf.transpose(tmp_f, perm=[1, 0, 2])
      inputs = cache["f"] = inputs[:, -first_kernel_size:, :]

    h = tpu_conv1d(
        inputs, filter_size, first_kernel_size, padding=padding, name="conv1")

    if cache:
      h = h[:, -1:, :]

    h = tf.nn.relu(h)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    return tpu_conv1d(
        h, output_size, second_kernel_size, padding=padding, name="conv2")


def sepconv_relu_sepconv(inputs,
                         filter_size,
                         output_size,
                         first_kernel_size=(1, 1),
                         second_kernel_size=(1, 1),
                         padding="LEFT",
                         nonpadding_mask=None,
                         dropout=0.0,
                         name=None):
  """可分离卷积 + ReLU + 可分离卷积的前馈层实现。
  
  与 dense_relu_dense 和 conv_relu_conv 类似，但使用深度可分离卷积，
  参数更少、计算更高效。默认使用 LEFT padding（因果卷积）。
  
  操作流程：写可分离卷积 -> ReLU -> Dropout -> 可分离卷积
  
  Args:
    inputs: 输入张量（支持 3D 和 4D）
    filter_size: FFN 隐藏层大小
    output_size: FFN 输出大小
    first_kernel_size: 第一个可分离卷积的卷积核大小
    second_kernel_size: 第二个可分离卷积的卷积核大小
    padding: 填充方式（默认 LEFT，适合自回归语言模型）
    nonpadding_mask: 非填啂位置的浮点掩码
    dropout: Dropout 率
    name: 变量作用域名称
    
  Returns:
    FFN 输出张量
  """
  with tf.variable_scope(name, "sepconv_relu_sepconv", [inputs]):
    inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    h = separable_conv(
        inputs,
        filter_size,
        first_kernel_size,
        activation=tf.nn.relu,
        padding=padding,
        name="conv1")
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    ret = separable_conv(
        h, output_size, second_kernel_size, padding=padding, name="conv2")
    if is_3d:
      ret = tf.squeeze(ret, 2)
    return ret


# DEPRECATED - 请使用 dense_relu_dense、conv_relu_conv 或 sepconv_relu_sepconv
def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     dropout=0.0,
                     **kwargs):
  """已废弃：卷积 + ReLU 激活 + 线性射影（请使用 dense_relu_dense 或 conv_relu_conv）。
  
  与 dense_relu_dense 类似，但使用卷积层。当 kernel_size=(1,1) 时等价于全连接。
  """
  name = kwargs.pop("name") if "name" in kwargs else None
  with tf.variable_scope(name, "conv_hidden_relu", [inputs]):
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    conv_f1 = conv if kernel_size == (1, 1) else separable_conv
    h = conv_f1(
        inputs,
        hidden_size,
        kernel_size,
        activation=tf.nn.relu,
        name="conv1",
        **kwargs)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    conv_f2 = conv if second_kernel_size == (1, 1) else separable_conv
    ret = conv_f2(h, output_size, second_kernel_size, name="conv2", **kwargs)
    if is_3d:
      ret = tf.squeeze(ret, 2)
    return ret


def conv_gru(x,
             kernel_size,
             filters,
             padding="SAME",
             dilation_rate=(1, 1),
             name=None,
             reuse=None):
  """卷积 GRU（门控循环单元），将卷积用于序列处理。
  
  GRU（Gated Recurrent Unit）是 LSTM 的简化版本，只有重置门和更新门。
  卷积 GRU 使用卷积替代全连接，可以利用空间局部性。
  
  实现：
  - reset gate: r = saturating_sigmoid(conv(x)) ：控制应保留多少历史状态
  - update gate: g = saturating_sigmoid(conv(x)) ：控制新状态的更新量
  - candidate: c = tanh(conv(r * x)) ：候选新状态
  - output = g * x + (1 - g) * c ：加权合并历史和候选状态
  
  Args:
    x: 输入张量 [batch, length, ...]
    kernel_size: 卷积核大小
    filters: 卷积输出通道数
    padding: 填充方式
    dilation_rate: 膨胀率
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    GRU 输出张量，形状与 x 相同
  """

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start, padding):
    return conv(
        args,
        filters,
        kernel_size,
        padding=padding,
        dilation_rate=dilation_rate,
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  with tf.variable_scope(
      name, default_name="conv_gru", values=[x], reuse=reuse):
    reset = saturating_sigmoid(do_conv(x, "reset", 1.0, padding))
    gate = saturating_sigmoid(do_conv(x, "gate", 1.0, padding))
    candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0, padding))
    return gate * x + (1 - gate) * candidate


def gru_feedfwd(a_t, h_prev, filters, name=None):
  """position-wise Feed-fwd GRU gates following the MPNN.

  Args:
    a_t: Tensor of shape [batch, length, depth] of current input
    h_prev: Tensor of shape [batch, length, depth] of prev input
    filters: an integer specifying number of dimensions of the filters
    name: A string
  Returns:
    h_t: [batch, length, filters] hidden state
  """

  with tf.variable_scope(name, default_name="GRU", values=[a_t, h_prev]):
    # we use right matrix multiplication to handle batches
    # W_z and W_r have shape 2d, d. U_z U_r have shape d,d
    z_t = (
        tf.sigmoid(
            tpu_conv1d(a_t, filters, 1, padding="SAME", name="W_z") +
            tpu_conv1d(h_prev, filters, 1, padding="SAME", name="U_z")))
    r_t = (
        tf.sigmoid(
            tpu_conv1d(a_t, filters, 1, padding="SAME", name="W_r") +
            tpu_conv1d(h_prev, filters, 1, padding="SAME", name="U_r")))
    h_tilde = (
        tf.tanh(
            tpu_conv1d(a_t, filters, 1, padding="SAME", name="W") +
            tpu_conv1d(r_t * h_prev, filters, 1, padding="SAME", name="U")))
    h_t = (1. - z_t) * h_prev + z_t * h_tilde

  return h_t


def conv_lstm(x,
              kernel_size,
              filters,
              padding="SAME",
              dilation_rate=(1, 1),
              name=None,
              reuse=None):
  """卷积 LSTM（长短期记忆单元），将卷积用于序列处理。
  
  LSTM 有 4 个门：输入门、遗忘门、输出门和候选内容，分别控制信息流。
  卷积 LSTM 使用卷积计算这 4 个门，利用空间局部性。
  
  实现（简化版）：
  - gates = conv(x, 4*filters)：一次卷积计算出所有门
  - 层归一化后分割为 4 组
  - new_cell = sigmoid(g[0]) * x + sigmoid(g[1]) * tanh(g[3])  # 单元状态更新
  - output = sigmoid(g[2]) * tanh(new_cell)  # 输出门
  
  Args:
    x: 输入张量 [batch, length, 1, depth]
    kernel_size: 卷积核大小
    filters: 卷积输出通道数
    padding: 填充方式
    dilation_rate: 膨胀率
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    LSTM 输出张量。注意：这是无状态的单步 LSTM，不维护本地单元状态
  """
  with tf.variable_scope(
      name, default_name="conv_lstm", values=[x], reuse=reuse):
    gates = conv(
        x,
        4 * filters,
        kernel_size,
        padding=padding,
        dilation_rate=dilation_rate)
    g = tf.split(layer_norm(gates, 4 * filters), 4, axis=3)
    new_cell = tf.sigmoid(g[0]) * x + tf.sigmoid(g[1]) * tf.tanh(g[3])
    return tf.sigmoid(g[2]) * tf.tanh(new_cell)


def diagonal_conv_gru(x,
                      kernel_size,
                      filters,
                      dropout=0.0,
                      name=None,
                      reuse=None):
  """对角卷积 GRU，参见论文 https://arxiv.org/abs/1702.08727。
  
  在标准卷积 GRU 基础上增加了对角移动（Diagonal Shift）：
  - 将 filters 个通道分为 3 组：中同、左移、右移
  - 通过深度卷积将不同组的通道分别往不同方向移动
  - 这使得网络能表示更丰富的时间局次关系
  
  使用硬 sigmoid 门函数（hard_sigmoid）代替饱和 sigmoid，计算更高效。
  
  Args:
    x: 输入张量
    kernel_size: 卷积核大小
    filters: 卷积输出通道数
    dropout: Dropout 率应用于候选状态
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    (output, total_cost_avg): 输出张量和门饱和惩罚平均値
  """

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start):
    return conv(
        args,
        filters,
        kernel_size,
        padding="SAME",
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  with tf.variable_scope(
      name, default_name="diagonal_conv_gru", values=[x], reuse=reuse):
    reset, reset_cost = hard_sigmoid(do_conv(x, "reset", 0.5))
    gate, gate_cost = hard_sigmoid(do_conv(x, "gate", 0.7))
    candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0))

    if dropout > 0.0:
      candidate = tf.nn.dropout(candidate, 1.0 - dropout)

    # Diagonal shift.
    shift_filters = filters // 3
    base_filter = ([[0, 1, 0]] * (filters - 2 * shift_filters) +
                   [[1, 0, 0]] * shift_filters + [[0, 0, 1]] * shift_filters)
    shift_filter = tf.constant(np.transpose(base_filter), dtype=tf.float32)
    shift_filter = tf.expand_dims(tf.expand_dims(shift_filter, 0), 3)
    x_shifted = tf.nn.depthwise_conv2d(
        x, shift_filter, [1, 1, 1, 1], padding="SAME")

    # Return the gated result and cost.
    total_cost_avg = 0.5 * (reset_cost + gate_cost)
    return gate * x_shifted + (1 - gate) * candidate, total_cost_avg


def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
  """将张量 x 和 y 在指定轴填啂使它们长度相同。
  
  常用于计算损失时对齐 logits 和 labels 的长度。
  
  Args:
    x: 张量
    y: 张量
    final_length_divisible_by: 填啂后的长度必须能被此数整除（用于向量化操作）
    axis: 填啂的维度，1 或 2
    
  Returns:
    (padded_x, padded_y): 填啂后的两个张量，它们在指定轴上长度相同
  """
  if axis not in [1, 2]:
    raise ValueError("Only axis=1 and axis=2 supported for now.")
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = shape_list(x)[axis]
    y_length = shape_list(y)[axis]
    if (isinstance(x_length, int) and isinstance(y_length, int) and
        x_length == y_length and final_length_divisible_by == 1):
      return x, y
    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by
    length_diff1 = max_length - x_length
    length_diff2 = max_length - y_length

    def padding_list(length_diff, arg):
      if axis == 1:
        return [[[0, 0], [0, length_diff]],
                tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
      return [[[0, 0], [0, 0], [0, length_diff]],
              tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

    paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
    paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
    res_x = tf.pad(x, paddings1)
    res_y = tf.pad(y, paddings2)
    # Static shapes are the same except for axis=1.
    x_shape = x.shape.as_list()
    x_shape[axis] = None
    res_x.set_shape(x_shape)
    y_shape = y.shape.as_list()
    y_shape[axis] = None
    res_y.set_shape(y_shape)
    return res_x, res_y


def pad_with_zeros(logits, labels):
  """将 labels 在长度维度上填啂零以匹配 logits 长度。
  
  计算损失时确保 logits 和 labels 的序列长度一致。
  对于 2D labels，同时在 axis=1 和 axis=2 上对齐。
  """
  with tf.name_scope("pad_with_zeros", values=[logits, labels]):
    logits, labels = pad_to_same_length(logits, labels)
    if len(labels.shape) == 3:  # 2-d labels.
      logits, labels = pad_to_same_length(logits, labels, axis=2)
    return logits, labels


def weights_nonzero(labels):
  """为所有非零标签分配权重 1.0，填啂 token（id=0）的权重为 0.0。
  
  最常用的损失权重函数：确保损失计算中填啂 token 不贡献。
  """
  return to_float(tf.not_equal(labels, 0))


def weights_prepend_inputs_to_targets(labels):
  """在 prepend_mode 中，为标签的 "targets" 部分分配权重 1.0。

  在 prepend 模式中，输入和输出拼接在一起：[source tokens] [0] [target tokens]
  其中 0 是分隔符。这个函数为第一个零之后的所有非零标签分配权重 1.0。

  Args:
    labels: int32 类型的张量

  Returns:
    浮点型权重张量
  """
  past_first_zero = tf.cumsum(to_float(tf.equal(labels, 0)), axis=1)
  nonzero = to_float(labels)
  return to_float(tf.not_equal(past_first_zero * nonzero, 0))


def check_nonnegative(value):
  """检查值是否为非负数（支持张量和 Python 数字）。
  
  Args:
    value: 要检查的张量或 Python 数字
    
  Returns:
    value 本身（加了检查依赖的张量）
    
  Raises:
    ValueError: 如果 value 是 Python 数字且小于 0
  """
  if isinstance(value, tf.Tensor):
    with tf.control_dependencies([tf.assert_greater_equal(value, 0)]):
      value = tf.identity(value)
  elif value < 0:
    raise ValueError("Value must be non-negative.")
  return value


def weights_multi_problem(labels, taskid=-1):
  """多任务损失权重：为标签序列中 taskid 后面的部分分配权重 1.0。

  在多任务训练尾巴式设置中，序列格式为：
  [source tokens] [taskid] [target tokens]
  其中 taskid 是任务标识符。这个函数为 taskid 后面的 target tokens 分配权重 1.0。

  Args:
    labels: int32 类型的张量
    taskid: 任务标识符的整数 ID

  Returns:
    浮点型权重张量

  Raises:
    ValueError: taskid 必须是有效的非负整数
  """
  taskid = check_nonnegative(taskid)
  past_taskid = tf.cumsum(to_float(tf.equal(labels, taskid)), axis=1)
  # Additionally zero out the task id location
  past_taskid *= to_float(tf.not_equal(labels, taskid))
  non_taskid = to_float(labels)
  return to_float(tf.not_equal(past_taskid * non_taskid, 0))


def weights_multi_problem_all(labels, taskid=-1):
  """多任务全程权重：对给定任务的整个示例（输入+输出）分配权重 1.0。
  
  与 weights_multi_problem 的区别：后者只对 target 部分分配权重，
  而这个函数对包含该 taskid 的整个示例（包括 source 和 target）分配权重。
  """
  taskid = check_nonnegative(taskid)
  weights = to_float(tf.not_equal(labels, 0))
  past_taskid = tf.cumsum(to_float(tf.equal(labels, taskid)), axis=1)
  # Additionally zero out the task id location
  past_taskid *= to_float(tf.not_equal(labels, taskid))
  non_taskid = to_float(labels)
  example_mask = to_float(tf.not_equal(past_taskid * non_taskid, 0))
  example_mask = tf.reduce_sum(example_mask, axis=1)
  example_mask = to_float(
      tf.greater(example_mask, tf.zeros_like(example_mask)))

  return weights * tf.expand_dims(example_mask, axis=-1)


def weights_multi_problem_input(labels, taskid=-1):
  """多任务输入权重：只对给定任务的输入部分分配权重 1.0。
  
  与 weights_multi_problem 相反：仅对 taskid 前面的 source tokens 分配权重。
  """
  taskid = check_nonnegative(taskid)
  weights_all_tokens = weights_multi_problem_all(labels, taskid)
  weights_target = weights_multi_problem(labels, taskid)
  return weights_all_tokens - weights_target


def weights_all(labels):
  """为所有标签分配权重 1.0（包括填啂 token）。
  
  与 weights_nonzero 不同，这个函数不跳过填啂 token，对所有位置计算损失。
  """
  return tf.ones_like(labels, dtype=tf.float32)


def weights_concatenated(labels):
  """为拼接标签的 "target" 部分分配权重 1.0（多句对训练时使用）。

  标签的格式如下：
    source English I love you . ID1 target French Je t'aime . ID1 source
      English the cat ID1 target French le chat ID1 source English ...

  其中 ID1 是句对分隔符。我们希望为目标文本的所有单词（包括 ID1 结尾符）
  分配权重 1.0，但不包括源语言文本和模板。上面的例子中，获得正权重的目标单词是：
    Je t'aime . ID1 le chat ID1

  实现方式：
  1. 通过 EOS (ID=1) 计算句号，奇数句编号 = target
  2. 排除每句头两个模板 token

  Args:
    labels: 标签张量
  Returns:
    浮点型权重张量
  """
  eos_mask = tf.to_int32(tf.equal(labels, 1))
  sentence_num = tf.cumsum(eos_mask, axis=1, exclusive=True)
  in_target = tf.equal(tf.mod(sentence_num, 2), 1)
  # first two tokens of each sentence are boilerplate.
  sentence_num_plus_one = sentence_num + 1
  shifted = tf.pad(sentence_num_plus_one,
                   [[0, 0], [2, 0], [0, 0], [0, 0]])[:, :-2, :, :]
  nonboilerplate = tf.equal(sentence_num_plus_one, shifted)
  ret = to_float(tf.logical_and(nonboilerplate, in_target))
  return ret


def padded_cross_entropy(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True,
                         cutoff=0.0,
                         gaussian=False):
  """计算序列分类任务的小垫交叉熵损失（忽略填啂 token）。

  计算损失分子（损失对加和）和损失分母（有效 token 数）。
  分母/分子的商是每 token 的平均损失。

  支持标签平滑（label smoothing）：
  给小部分概率了其他类别，防止模型对训练数据过度自信，改善泛化。

  Args:
    logits: 形状 [batch, timesteps, vocab_size] 的张量，或 FactoredTensor
    labels: 形状 [batch, timesteps] 的整数张量
    label_smoothing: 标签平滑系数（0 表示不平滑，0.1 是常用值）
    weights_fn: 将标签映射到权重的函数（默认忽略填啂 token）
    reduce_sum: 是否返回标量级损失（True）还是分位置损失（False）
    cutoff: 此值以下的损失不计入（通过 ReLU 切雴）
    gaussian: 如果为 True，使用高斯分布进行标签平滑

  Returns:
    loss_numerator: 标量，损失对加和。
    loss_denominator: 标量，非填啂目标 token 的数量。

  Raises:
    ValueError: 当参数类型不支持时
  """
  if isinstance(logits, FactoredTensor):
    if gaussian:
      raise ValueError("Factored padded cross entropy with Gaussian smoothing "
                       "is not implemented yet.")
    return padded_cross_entropy_factored(
        logits,
        labels,
        label_smoothing,
        weights_fn=weights_fn,
        reduce_sum=reduce_sum)
  confidence = 1.0 - label_smoothing
  logits_shape = shape_list(logits)
  vocab_size = logits_shape[-1]
  with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
    if len(logits_shape) == 2:
      # Deal with the case where we did not insert extra dimensions due to
      # TPU issues.  No pad-to-same-length happens in this case.
      # TODO(noam): remove this logic once TPU can handle extra dimensions.
      labels = tf.reshape(labels, [-1])
    else:
      logits, labels = pad_with_zeros(logits, labels)
    logits = tf.reshape(
        logits,
        shape_list(labels) + [vocab_size],
        name="padded_cross_entropy_size_check")
    logits = tf.cast(logits, tf.float32)
    xent = smoothing_cross_entropy(
        logits, labels, vocab_size, confidence, gaussian=gaussian)
    weights = weights_fn(labels)
    if cutoff > 0.0:
      xent = tf.nn.relu(xent - cutoff)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def _weights_one_third(labels):
  """返回形状为 [batch, height, width] 的张量，每个元素为 1/3。
  
  用于 RGB 图像损失计算：将 3 通道的损失均分权重以计算每像素平均损失。
  """
  return tf.ones(tf.shape(labels)[:-1]) / 3.


def dml_loss(pred, labels, weights_fn=_weights_one_third, reduce_sum=True):
  """离散化混合逻辑斯分布损失（Discretized Mixture of Logistics Loss）。
  
  用于图像生成模型（如 PixelCNN）的像素级损失函数。
  将每个像素的 RGB 条件分布建as 多个离散化逻辑斯分布的混合，
  能够表达多模的像素分布。

  Args:
    pred: 形状 [batch, height, width, num_mixtures*10] 的浮点张量
      包括：一个未约束混合概率、三个均値（每通道一个）、三个标准差、
      三个跨通道线性依赖系数
    labels: 形状 [batch, height, width, channels] 的 8位像素张量（假定 channels=3）
    weights_fn: 将标签映射到 [batch, height, width] 权重的函数。
      默认将每个损失项缩放 1/3，捕获通道间的平均
    reduce_sum: 是否返回标量损失而非每位置损失

  Returns:
    (loss_num, loss_den) 元组，分别是损失分子和分母
    当 reduce_sum=True 时为标量，否则为 [batch, height, width] 形状
    二者之商表示 labels 中每像素的 nats 数（信息量）
  """
  real_labels = convert_rgb_to_symmetric_real(labels)
  dml_loss_value = discretized_mix_logistic_loss(pred=pred, labels=real_labels)
  weights = weights_fn(labels)
  loss_num = weights * dml_loss_value
  loss_den = weights_nonzero(weights)
  if reduce_sum:
    loss_num = tf.reduce_sum(loss_num)
    loss_den = tf.reduce_sum(loss_den)
  return loss_num, loss_den


def split_to_discretized_mix_logistic_params(inputs):
  """将输入张量分割成离散化混合逻辑斯分布的各个参数张量。

  Args:
    inputs: 形状 [batch, height, width, num_mixtures*10] 的浮点张量
      包括：一个未约束的混合概率、三个均値（每通道）、三个标准差，
      以及三个跨通道线性依赖系数

  Returns:
    (混合概率, 均値, 对数标准差, 系数) 元组。
    混合概率形状：[batch, height, width, num_mixtures]
    其他参数形状：[batch, height, width, num_mixtures, 3]
  """
  batch, height, width, output_dim = shape_list(inputs)  # pylint: disable=unbalanced-tuple-unpacking
  num_mixtures = output_dim // 10
  logits, locs, log_scales, coeffs = tf.split(
      inputs,
      num_or_size_splits=[
          num_mixtures, num_mixtures * 3, num_mixtures * 3, num_mixtures * 3
      ],
      axis=-1)
  split_shape = [batch, height, width, num_mixtures, 3]
  locs = tf.reshape(locs, split_shape)
  log_scales = tf.reshape(log_scales, split_shape)
  log_scales = tf.maximum(log_scales, -7.)
  coeffs = tf.reshape(coeffs, split_shape)
  coeffs = tf.tanh(coeffs)
  return logits, locs, log_scales, coeffs


def discretized_mix_logistic_loss(pred, labels):
  """Computes negative log probability for the discretized mixture of logistics.

  The distribution of a whole pixel is a mixture of 3-dimensional discretized
  logistic distributions. The 3-D discretized logistic factorizes as 3 1-D
  discretized logistic distributions, one for each channel. It defines

  ```none
  P(X = x)
  = sum_{k=1}^K probs[k] * P(X = x | locs[k], scales[k])
  = sum_{k=1}^K probs[k] * [
      prod_{c=1}^3 DiscretizedLogistic(X[c] = x[c] | means[k][c], scales[k]) ]
  ```

  The means tensor is a linear combination of location parameters and previous
  channels. The discretized logistic distribution assigns probability mass to an
  event P(X=x) via logistic CDFs: P(X <= x + 0.5) - P(X < x - 0.5) for 1 < x <
  254; P(X <= 0.5) for x = 0; and 1 - P(X < 245.5) for x = 255. Instead of
  8-bit inputs, this implementation assumes the events are rescaled to [-1, 1].

  Args:
    pred: A [batch, height, width, num_mixtures*10] tensor of floats
      comprising one unconstrained mixture probability, three means
      (one per channel), three standard deviations (one per channel),
      and three coefficients which linearly parameterize dependence across
      channels.
    labels: A [batch, height, width, channels] tensor of true pixel intensities
      rescaled to [-1, 1]. The computation assumes channels is 3.

  Returns:
    A [batch, height, width] tensor of the negative log conditional probability
    of each pixel given all previous pixels.
  """

  logits, locs, log_scales, coeffs = split_to_discretized_mix_logistic_params(
      pred)

  # Tile labels to broadcast compute across the mixture dimension.
  batch, height, width, num_mixtures = shape_list(logits)  # pylint: disable=unbalanced-tuple-unpacking
  labels = tf.tile(
      tf.reshape(labels, [batch, height, width, 1, 3]),
      [1, 1, 1, num_mixtures, 1])

  # p(x) = sigmoid((x - means_i + 1/255.)/scale_i) -
  #        sigmoid((x - means_i - 1/255.)/scale_i)
  # for each channel i. The means are linearly parameterized.
  means_0 = locs[..., 0]
  means_1 = locs[..., 1] + coeffs[..., 0] * labels[..., 0]
  means_2 = (
      locs[..., 2] + coeffs[..., 1] * labels[..., 0] +
      coeffs[..., 2] * labels[..., 1])
  means = tf.stack([means_0, means_1, means_2], axis=-1)
  centered_labels = labels - means
  inv_stdv = tf.exp(-log_scales)
  plus_in = inv_stdv * (centered_labels + 1. / 255.)
  min_in = inv_stdv * (centered_labels - 1. / 255.)
  cdf_plus = tf.nn.sigmoid(plus_in)
  cdf_min = tf.nn.sigmoid(min_in)

  # Compute log probability for edge case of 0 (before scaling), 255 (before
  # scaling), and all other cases respectively.
  log_prob_0 = plus_in - tf.nn.softplus(plus_in)
  log_prob_255 = -tf.nn.softplus(min_in)
  prob_event = tf.maximum(cdf_plus - cdf_min, 1e-12)
  log_prob_event = tf.log(prob_event)

  # Robustly select log-prob based on numerical edge-cases: (a) [-1, -1+eps);
  # (b) (1-eps, 1]; (c) NaNs during `tf.gradients` of `tf.select`, which may
  # cause `tf.log(0.)`; (d) p(x) < 1e-5.
  mid_in = inv_stdv * centered_labels
  log_prob_event_approx = (
      mid_in - log_scales - 2. * tf.nn.softplus(mid_in) - np.log(127.5))
  log_probs = tf.where(
      labels < -0.999, log_prob_0,
      tf.where(
          labels > 0.999, log_prob_255,
          tf.where(prob_event > 1e-5, log_prob_event, log_prob_event_approx)))

  # Sum over channels and compute log-probability of each mixture.
  log_probs = tf.reduce_sum(log_probs, -1) + tf.nn.log_softmax(logits, axis=-1)
  output = -tf.reduce_logsumexp(log_probs, axis=-1)
  return output


def sample_from_discretized_mix_logistic(pred, seed=None):
  """从离散化混合逻辑斯分布中采样（用于图像生成）。

  推理时用于根据模型输出的参数生成新像素。
  实现步骤：
  1. 使用 Gumbel-Max 技巧采样混合分量索引
  2. 选择对应分量的参数（均値、标准差、系数）
  3. 从 3D 逻辑斯分布中采样（通过等价均匀采样转换）
  4. 使用系数对通道间进行线性依赖校正
  5. 输出截断到 [-1, 1]

  Args:
    pred: 形状 [batch, height, width, num_mixtures*10] 的浮点张量
      包括：一个未约束混合概率、三个均値（每通道一个）、三个标准差，
      以及三个跨通道线性依赖系数
    seed: 随机数种子

  Returns:
    形状为 [batch, height, width, 3] 的张量，像素强度缩放至 [-1, 1]
  """

  logits, locs, log_scales, coeffs = split_to_discretized_mix_logistic_params(
      pred)

  # Sample mixture indicator given logits using the gumbel max trick.
  num_mixtures = shape_list(logits)[-1]
  gumbel_noise = -tf.log(-tf.log(
      tf.random_uniform(
          tf.shape(logits), minval=1e-5, maxval=1. - 1e-5, seed=seed)))
  sel = tf.one_hot(
      tf.argmax(logits + gumbel_noise, -1),
      depth=num_mixtures,
      dtype=tf.float32)

  # Select mixture component's parameters.
  sel = tf.expand_dims(sel, -1)
  locs = tf.reduce_sum(locs * sel, 3)
  log_scales = tf.reduce_sum(log_scales * sel, 3)
  coeffs = tf.reduce_sum(coeffs * sel, 3)

  # Sample from 3-D logistic & clip to interval. Note we don't round to the
  # nearest 8-bit value when sampling.
  uniform_noise = tf.random_uniform(
      tf.shape(locs), minval=1e-5, maxval=1. - 1e-5, seed=seed)
  logistic_noise = tf.log(uniform_noise) - tf.log1p(-uniform_noise)
  x = locs + tf.exp(log_scales) * logistic_noise
  x0 = x[..., 0]
  x1 = x[..., 1] + coeffs[..., 0] * x0
  x2 = x[..., 2] + coeffs[..., 1] * x0 + coeffs[..., 2] * x1
  x = tf.stack([x0, x1, x2], axis=-1)
  x = tf.clip_by_value(x, -1., 1.)
  return x


def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False):
  """带标签平滑的交叉熵，降低模型过度自信。

  标签平滑（Label Smoothing）是一种正则化技术：
  - 不是对留实类别使用 one-hot（信心 confidence=1.0）
  - 而是将少量概率分配给其他类别（每个类别 low_confidence）
  - 这样可以防止模型对训练标签过度自信，改善泛化

  Args:
    logits: 形状 [batch_size, ?, ?, ?, vocab_size] 的张量
    labels: 形状 [batch_size, ?, ?, ?] 的张量
    vocab_size: 词表大小（张量或整数）
    confidence: 留实类别的概率。
      如果 gaussian=True，则表示高斯分布的方差
    gaussian: 是否使用高斯分布进行标签平滑

  Returns:
    形状 [batch_size, ?, ?, ?] 的交叉熵张量
  """
  with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
        confidence * tf.log(confidence) + to_float(vocab_size - 1) *
        low_confidence * tf.log(low_confidence + 1e-20))

    if gaussian and confidence > 0.0:
      labels = tf.cast(labels, tf.float32)

      normal_dist = tfp.distributions.Normal(loc=labels, scale=confidence)
      # Locations to evaluate the probability distributions.
      soft_targets = normal_dist.prob(
          tf.cast(tf.range(vocab_size), tf.float32)[:, None, None, None, None])
      # Reordering soft_targets from [vocab_size, batch_size, ?, ?, ?] to match
      # logits: [batch_size, ?, ?, ?, vocab_size]
      soft_targets = tf.transpose(soft_targets, perm=[1, 2, 3, 4, 0])
    else:
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing


def global_pool_1d(inputs, pooling_type="MAX", mask=None):
  """在序列维度上进行全局池化，将序列转换为单个向量。

  用于将向量列表转换为单个向量，从而获得集合的表示。
  常用于集合分类任务，将可变长度序列压缩为固定大小的表示向量。

  Args:
    inputs: 形状 [batch_size, sequence_length, input_dims] 的张量
    pooling_type: 池化类型，"MAX"（最大池化）或 "AVR"（平均池化）
    mask: 形状 [batch_size, sequence_length] 的浮点掩码张量，
      1 表示有效元素，0 表示填啂

  Returns:
    形状 [batch_size, input_dims] 的张量，序列的全局表示
  """
  with tf.name_scope("global_pool", values=[inputs]):
    if mask is not None:
      mask = tf.expand_dims(mask, axis=2)
      inputs = tf.multiply(inputs, mask)

    if pooling_type == "MAX":
      # A tf.pool can be used here, but reduce is cleaner
      output = tf.reduce_max(inputs, axis=1)
    elif pooling_type == "AVR":
      if mask is not None:
        # Some elems are dummy elems so we can't just reduce the average.
        output = tf.reduce_sum(inputs, axis=1)
        num_elems = tf.reduce_sum(mask, axis=1, keepdims=True)
        output = tf.div(output, tf.maximum(num_elems, 1))
      else:
        output = tf.reduce_mean(inputs, axis=1)

  return output


def running_global_pool_1d(inputs, pooling_type="MAX"):
  """按序列累积进行全局池化（每个位置只看到它前面的元素）。

  与 global_pool_1d 相同，但每个位置只对该位置之前的元素进行池化。
  适用于未来状态不得而知的输出实现（因果池化）。
  假定从开始到当前位置的所有元素均有效（无需掩码）。
  等价于使用下三角偏置的注意力模式。
  目前只支持最大池化。

  Args:
    inputs: 形状 [batch_size, sequence_length, input_dims] 的张量
    pooling_type: 池化类型，目前只支持 'MAX'。

  Returns:
    A tensor of shape [batch_size, sequence_length, input_dims] containing the
    running 'totals'.
  """
  del pooling_type
  with tf.name_scope("running_global_pool", values=[inputs]):
    scan_fct = tf.maximum
    # Permute inputs so seq_length is first.
    elems = tf.transpose(inputs, [1, 0, 2])
    # Perform scan.
    cumulatives = tf.scan(scan_fct, elems, swap_memory=True)
    # Permute output to get back to original order.
    output = tf.transpose(cumulatives, [1, 0, 2])
  return output


def gated_linear_unit_layer(x, name=None):
  """门控线性单元（GLU）层实现。

  论文：Language Modeling with Gated Convolutional Networks
  链接：https://arxiv.org/abs/1612.08083
  
  实现公式：x_new = Wx * sigmoid(W'x)
  将输入分为两部分：内容部分和门控部分。
  内容部分经过 sigmoid 门控的过滤。
  
  GLU 是全连接层和 sigmoid 激活的组合，对内容进行选择性开可，
  类似于 GRU 中的门控机制。

  Args:
    x: 张量输入
    name: 变量作用域名称

  Returns:
    与 x 形状相同的张量
  """
  with tf.variable_scope(name, default_name="glu_layer", values=[x]):
    depth = shape_list(x)[-1]
    x = layers().Dense(depth * 2, activation=None)(x)
    x, gating_x = tf.split(x, 2, axis=-1)
    return x * tf.nn.sigmoid(gating_x)


def sru(x,
        num_layers=2,
        activation=None,
        initial_state=None,
        name=None,
        reuse=None):
  """简单循环单元（SRU），参见论文 https://arxiv.org/abs/1709.02755。

  SRU（Simple Recurrent Unit）是一种循环神经网络单元，设计为容易并行化：
  - 将线性变换和状态转移分离，线性变换可并行计算
  - 状态转移不依赖上一时间步的状态，可用 tf.scan 高效实现
  
  对于每层：
  - 并行计算：[x, f, r] = Dense(x_input)
    - x：内容向量
    - f：遗忘门（forget gate）
    - r：高速连接门（highway gate）
  - 递归计算：c[t] = f[t] * c[t-1] + (1-f[t]) * x[t] (tf.scan)
  - 输出：h[t] = r[t] * c[t] + (1-r[t]) * x_orig[t]
  
  这个实现使用 tf.scan，有一定开销，另见完整版本的文档获取更快实现。

  Args:
    x: 形状 [batch, ..., channels] 的张量；... 被视为时间维度
    num_layers: SRU 层数，默认为 2（单层效果较差）
    activation: 可选的激活函数，可尝试 tf.nn.tanh 或 tf.nn.relu
    initial_state: 可选的初始 c 状态，None 时初始化为零
    name: 可选的名称，默认 "sru"
    reuse: 可选的重用标志

  Returns:
    与 x 形状相同的张量

  Raises:
    ValueError: 如果 num_layers 不是正整数
  """
  if num_layers < 1:
    raise ValueError("Number of layers must be positive: %d" % num_layers)
  with tf.variable_scope(name, default_name="sru", values=[x], reuse=reuse):
    # We assume x is [batch, ..., channels] and treat all ... as time.
    x_shape = shape_list(x)
    x = tf.reshape(x, [x_shape[0], -1, x_shape[-1]])
    x = tf.transpose(x, [1, 0, 2])  # Scan assumes time on axis 0.
    initial_state = initial_state or tf.zeros([x_shape[0], x_shape[-1]])

    # SRU state manipulation function.
    def next_state(cur_state, args_tup):
      cur_x_times_one_minus_f, cur_f = args_tup
      return cur_f * cur_state + cur_x_times_one_minus_f

    # Calculate SRU on each layer.
    for i in range(num_layers):
      # The parallel part of the SRU.
      x_orig = x
      x, f, r = tf.split(
          layers().Dense(3 * x_shape[-1], name="kernel_%d" % i)(x), 3, axis=-1)
      f, r = tf.sigmoid(f), tf.sigmoid(r)
      x_times_one_minus_f = x * (1.0 - f)  # Compute in parallel for speed.
      # Calculate states.
      c_states = tf.scan(
          next_state, (x_times_one_minus_f, f),
          initializer=initial_state,
          parallel_iterations=2,
          name="scan_%d" % i)
      # Final output.
      if activation is not None:
        c_states = activation(c_states)
      h = c_states * r + (1.0 - r) * x_orig
      x = h  # Next layer.
    # Transpose back to batch-major.
    x = tf.transpose(x, [1, 0, 2])
    return tf.reshape(x, x_shape)


def linear_set_layer(layer_size,
                     inputs,
                     context=None,
                     activation_fn=tf.nn.relu,
                     dropout=0.0,
                     name=None):
  """对集合每个元素应用线性变换的基本集合层。

  对输入集合中每个元素应用线性变换。
  如果提供了上下文，将其与输入混合（通过加法实现广播）。
  例：可用 global_pool_1d 获取集合表示，然后作为下一层的上下文。

  Args:
    layer_size: 输入向量变换的目标维度
    inputs: 形状 [batch_size, sequence_length, input_dims] 的张量
    context: 形状 [batch_size, context_dims] 的全局上下文张量
    activation_fn: 使用的激活函数
    dropout: Dropout 概率
    name: 变量作用域名称

  Returns:
    形状 [batch_size, sequence_length, output_dims] 的张量
  """
  with tf.variable_scope(
      name, default_name="linear_set_layer", values=[inputs]):
    # Apply 1D convolution to apply linear filter to each element
    # along the 2nd dimension.
    outputs = conv1d(inputs, layer_size, 1, activation=None, name="set_conv")

    # Apply the context if it exists.
    if context is not None:
      # Unfortunately tf doesn't support broadcasting via concat, but we can
      # simply add the transformed context to get the same effect.
      if len(context.get_shape().as_list()) == 2:
        context = tf.expand_dims(context, axis=1)
      cont_tfm = conv1d(
          context, layer_size, 1, activation=None, name="cont_conv")
      outputs += cont_tfm

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    if dropout != 0.0:
      outputs = tf.nn.dropout(outputs, 1.0 - dropout)

    return outputs


def ravanbakhsh_set_layer(layer_size,
                          inputs,
                          mask=None,
                          sequential=False,
                          activation_fn=tf.nn.tanh,
                          dropout=0.0,
                          name=None):
  """来自 Deep Sets 论文的集合层：https://arxiv.org/abs/1611.04500。

  带上下文的 linear_set_layer 的更高参数效率版本。
  
  Deep Sets 的核心思想：每个元素的输入减去整个集合的全局表示，
  让层学习元素与集合之间的差异。
  持丢序不变性，即集合的顺序不影响结果（如果 sequential=False）。

  Args:
    layer_size: 输入向量变换的目标维度
    inputs: 形状 [batch_size, sequence_length, vector] 的张量
    mask: 形状 [batch_size, sequence_length] 的掩码，1=有效，0=填啂
    sequential: 如果为 True，使用 running_global_pool ，
      使每个元素只依赖它前面的元素（用于自回归输出）
    activation_fn: 使用的激活函数
    dropout: Dropout 概率
    name: 变量作用域名称

  Returns:
    形状 [batch_size, sequence_length, vector] 的张量
  """
  del dropout
  with tf.variable_scope(name, "ravanbakhsh_set_layer", [inputs]):
    if sequential:
      return linear_set_layer(
          layer_size,
          inputs - running_global_pool_1d(inputs),
          activation_fn=activation_fn,
          name=name)
    return linear_set_layer(
        layer_size,
        inputs - tf.expand_dims(global_pool_1d(inputs, mask=mask), axis=1),
        activation_fn=activation_fn,
        name=name)


def fn_device_dependency_dict():
  """获取当前默认计算图的设备依赖字典（函数间跨设备同步的状态容器）。
  
  用于 fn_device_dependency 上下文管理器的内部实现。
  每个 name+device 组合对应一个控制依赖列表，确保同一设备上相同名称的操作按顺序执行。
  
  Returns:
    defaultdict：键是 name_device 字符串，值是控制依赖的 Tensor 列表
  """
  default_graph = tf.get_default_graph()
  if not hasattr(default_graph, "dependency_dict"):
    default_graph.dependency_dict = collections.defaultdict(list)
  return default_graph.dependency_dict


@contextlib.contextmanager
def fn_device_dependency(name, device=""):
  """添加同一名称和设备之间的控制依赖（确保顺序执行的上下文管理器）。
  
  用于确保同一设备上相同名称的操作不会乱序执行：
  - 下一次调用会等待上一次调用的输出就绪后再执行
  - 主要用于内存高效实现中分批次处理时保证顺序
  
  使用方式：
    with fn_device_dependency("my_fn", device="/gpu:0") as outs:
      outs[:] = [some_computation()]  # 输出必须写入 outs
  
  Args:
    name: 依赖组名称，相同名称的调用之间会有控制依赖
    device: 可选设备字符串（如 '/gpu:0'，'' 表示默认设备）
  """
  key = name + "_" + device
  outs = []

  def body():
    with tf.control_dependencies(fn_device_dependency_dict()[key]):
      yield outs
      assert outs

      deps = outs
      if isinstance(outs[0], (list, tuple)):
        assert len(outs) == 1
        deps = outs[0]
      fn_device_dependency_dict()[key] = deps

  if device:
    with tf.device(device):
      return body()
  else:
    return body()


def underlying_variable_ref(t):
  """查找张量对应的底层变量引用（穿透 Identity、ReadVariableOp、Enter 等操作）。

  用于追踪一个张量是否来自某个 tf.Variable：
  - 跳过 Identity（恒等变换）、ReadVariableOp（变量读取）、Enter（进入 while_loop）等透明操作
  - 找到包含 'Variable' 或 'VarHandle' 的 op 时停止
  - 如果无法找到变量引用，返回 None

  Args:
    t: 任意张量

  Returns:
    底层变量引用张量（Tensor），或者 None（如果 t 不是变量操作的结果）
  """
  while t.op.type in ["Identity", "ReadVariableOp", "Enter"]:
    t = t.op.inputs[0]

  op_type = t.op.type
  if "Variable" in op_type or "VarHandle" in op_type:
    return t
  else:
    return None


def underlying_variable(t):
  """找到张量对应的底层 tf.Variable 对象。

  Args:
    t: 张量

  Returns:
    对应的 tf.Variable 对象
  """
  t = underlying_variable_ref(t)
  assert t is not None
  # make sure that the graph has a variable index and that it is up-to-date
  if not hasattr(tf.get_default_graph(), "var_index"):
    tf.get_default_graph().var_index = {}
  var_index = tf.get_default_graph().var_index
  for v in tf.global_variables()[len(var_index):]:
    var_index[v.name] = v
  return var_index[t.name]


def approximate_split(x, num_splits, axis=0):
  """将张量尽量均分地分割为 num_splits 个分。

  当张量大小不能被 num_splits 整除时，分配余数到前第几分。
  用于将大张量分块计算，备内存。

  Args:
    x: 张量
    num_splits: 分割数量
    axis: 分割轴（默认 axis=0）

  Returns:
    包含 num_splits 个张量的列表
  """
  size = shape_list(x)[axis]
  size_splits = [tf.div(size + i, num_splits) for i in range(num_splits)]
  return tf.split(x, size_splits, axis=axis)


class FactoredTensor(object):
  """张量的分解表示：用两个张量的矩阵乘积表示大张量。

  这个类代表张量 tf.matmul(a, b, transpose_b=True)，
  通过存储张量 a 和 b 的值而不是直接计算完整乘积。

  设计原因：乘积张量可能太大一次性不能全部实现，
  可以分次少量实现。

  a 可能有额外的领头维度，在计算之前会先展平，然后在实现后重新扩展。
  """

  def __init__(self, a, b):
    self._a = a
    self._b = b

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  def to_tensor(self):
    """Convert to Tensor."""
    a_shape = shape_list(self.a)
    b_shape = shape_list(self.b)
    inner_dim = b_shape[1]
    result_dim = b_shape[0]
    flat_a = tf.reshape(self.a, [-1, inner_dim])
    product = tf.matmul(flat_a, self.b, transpose_b=True)
    product_shape = a_shape[:-1] + [result_dim]
    product = tf.reshape(product, product_shape)
    product.set_shape(self.a.get_shape().as_list()[:-1] +
                      [self.b.get_shape()[0]])
    return product


def _convert_factored_tensor_to_tensor(value, *args, **kwargs):
  # call ops.convert_to_tensor to handle optional arguments appropriately
  return ops.convert_to_tensor(value.to_tensor(), *args, **kwargs)


tf.register_tensor_conversion_function(FactoredTensor,
                                       _convert_factored_tensor_to_tensor)


def smoothing_cross_entropy_factored_grad(op, dy):
  """Gradient function for smoothing_cross_entropy_factored."""
  a = op.inputs[0]
  b = op.inputs[1]
  labels = op.inputs[2]
  confidence = op.inputs[3]
  num_splits = 16
  vocab_size = shape_list(b)[0]
  labels = approximate_split(labels, num_splits)
  a = approximate_split(a, num_splits)
  dy = approximate_split(dy, num_splits)
  b_grad = None
  a_grad_parts = []
  deps = []
  for part in range(num_splits):
    with tf.control_dependencies(deps):
      logits = tf.matmul(a[part], b, transpose_b=True)
      output_part = smoothing_cross_entropy(logits, labels[part], vocab_size,
                                            confidence)
      a_grad_part, b_grad_part = tf.gradients(
          ys=[output_part], xs=[a[part], b], grad_ys=[dy[part]])
      a_grad_parts.append(a_grad_part)
      if part > 0:
        b_grad += b_grad_part
      else:
        b_grad = b_grad_part
      deps = [b_grad, a_grad_part]
  a_grad = tf.concat(a_grad_parts, 0)
  return a_grad, b_grad, None, None


@function.Defun(
    noinline=True,
    python_grad_func=smoothing_cross_entropy_factored_grad,
    compiled=True,
    separate_compiled_gradients=True)
def smoothing_cross_entropy_factored(a, b, labels, confidence):
  """内存高效的平滑交叉熵计算，避免一次性实现完整 logits 矩阵。
  
  通过将 logits = matmul(a, b.T) 分批计算，每次只实现 logits 的一小部分，
  然后立即计算损失，避免内存溢出。

  Args:
    a: 形状 [batch, inner_dim] 的张量
    b: 形状 [vocab_size, inner_dim] 的张量（嵌入矩阵）
    labels: 形状 [batch] 的整数张量
    confidence: 浮点数（留实类别的概率）

  Returns:
    形状 [batch] 的张量
  """
  num_splits = 16
  vocab_size = shape_list(b)[0]
  labels = approximate_split(labels, num_splits)
  a = approximate_split(a, num_splits)
  parts = []
  for part in range(num_splits):
    with tf.control_dependencies(parts[-1:]):
      logits = tf.matmul(a[part], b, transpose_b=True)
      parts.append(
          smoothing_cross_entropy(logits, labels[part], vocab_size, confidence))
  return tf.concat(parts, 0)


def padded_cross_entropy_factored(factored_logits,
                                  labels,
                                  label_smoothing,
                                  weights_fn=weights_nonzero,
                                  reduce_sum=True):
  """内存高效的填啂交叉熵计算，攟持 FactoredTensor 输入。

  避免一次性实现完整 logits 矩阵。

  Args:
    factored_logits: 表示形状 [batch, timesteps, vocab_size] 的 FactoredTensor
    labels: 形状 [batch, timesteps] 的整数张量
    label_smoothing: 标签平滑系数
    weights_fn: 将标签映射到权重的函数
    reduce_sum: 是否返回标量损失

  Returns:
    loss_numerator: 标量，损失对加和
    loss_denominator: 标量，非填啂目标 token 的数量
  """
  a = factored_logits.a
  b = factored_logits.b
  confidence = 1.0 - label_smoothing
  with tf.name_scope("padded_cross_entropy_factored", values=[a, b, labels]):
    labels_flat = tf.reshape(labels, [-1])
    a_flat = tf.reshape(a, [-1, shape_list(b)[1]])
    xent = smoothing_cross_entropy_factored(a_flat, b, labels_flat,
                                            tf.convert_to_tensor(confidence))
    xent = tf.reshape(xent, shape_list(labels))
    weights = weights_fn(labels)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def fn_with_custom_grad(grad_fn, use_global_vars=False):
  """装饰器：创建带自定义梯度函数的子图。

  被装饰函数创建的子图不会放入 Defun 中，因此不受 Defun 的限制：
  - 子图操作可以在不同设备上
  - 支持求和操作（summaries）
  
  常用于实现梯度检置点（gradient checkpointing）或其他需要
  自定义梯度的场景（如使用近似梯度提高效率）。

  Args:
    grad_fn: function with signature
      (inputs, variables, outputs, output_grads) -> (grad_inputs, grad_vars),
      all of which are lists of Tensors.
    use_global_vars: if True, variables will be the global variables created.
      If False, will be the trainable variables.

  Returns:
    Decorator for function such that the gradient is defined by grad_fn.
  """

  def dec(fn):

    @functools.wraps(fn)
    def wrapped(*args):
      return _fn_with_custom_grad(
          fn, args, grad_fn, use_global_vars=use_global_vars)

    return wrapped

  return dec


def _fn_with_custom_grad(fn, inputs, grad_fn, use_global_vars=False):
  """Create a subgraph with a custom gradient.

  Args:
    fn: function that takes inputs as arguments and produces 1 or more Tensors.
    inputs: list<Tensor>, will be passed as fn(*inputs).
    grad_fn: function with signature
      (inputs, vars, outputs, output_grads) -> (grad_inputs, grad_vars),
      all of which are lists of Tensors.
    use_global_vars: if True, variables will be the global variables created.
      If False, will be the trainable variables.

  Returns:
    fn(*inputs)
  """
  vs = tf.get_variable_scope()
  get_vars_fn = (
      vs.global_variables if use_global_vars else vs.trainable_variables)
  len_before_vars = len(get_vars_fn())
  inputs = list(inputs)
  outputs = fn(*inputs)
  train_vars = get_vars_fn()[len_before_vars:]

  if grad_fn is None:
    return outputs

  if not isinstance(outputs, (tuple, list)):
    outputs = [outputs]
  outputs = list(outputs)

  defun_inputs = [inputs, train_vars, outputs]

  def custom_grad_fn(op, *dys):
    """Custom grad fn applying grad_fn for identity Defun."""
    fn_inputs, fn_vars, fn_outputs = contrib.framework().nest.pack_sequence_as(
        defun_inputs, list(op.inputs))
    dys = list(dys)
    assert len(fn_outputs) == len(outputs)
    assert len(fn_outputs) == len(dys)

    grad_inputs, grad_vars = grad_fn(fn_inputs, fn_vars, fn_outputs, dys)
    grad_outputs = [None] * len(fn_outputs)
    return tuple(grad_inputs + grad_vars + grad_outputs)

  # The Defun takes as input the original inputs, the trainable variables
  # created in fn, and the outputs. In the forward it passes through the
  # outputs. In the backwards, it produces gradients for the original inputs
  # and the trainable variables.
  in_types = [t.dtype for t in inputs]
  out_types = [t.dtype for t in outputs]
  var_types = [t.dtype for t in train_vars]

  @function.Defun(
      *(in_types + var_types + out_types),
      func_name="identity_custom_grad%d" % ops.uid(),
      python_grad_func=custom_grad_fn,
      shape_func=lambda _: [t.get_shape() for t in outputs])
  def identity(*args):
    _, _, outs = contrib.framework().nest.pack_sequence_as(defun_inputs, args)
    return tuple([tf.identity(t) for t in outs])

  flat_inputs = contrib.framework().nest.flatten(defun_inputs)
  id_out = identity(*flat_inputs)
  return id_out


_function_cache = {}


def conv_hidden_relu_memory_efficient(x,
                                      filter_size,
                                      epsilon=1e-6,
                                      forget=True,
                                      test_vars=None,
                                      name=None):
  """内存高效的：LayerNorm + Conv + ReLU + Conv，支持梯度检置点。

  所有卷积核大小为 1（等价于全连接）。

  返回 conv(relu(conv(layer_norm(x))))
  
  当 forget=True 时，使用梯度检置点（gradient checkpointing）：
  - 前向传播时不保留中间激活，节约内存
  - 反向传播时重新计算激活值（用时间换内存）

  Args:
    x: 形状 [batch, length, io_size] 的输入张量
    filter_size: 隐藏层大小（整数）
    epsilon: 层归一化的数字稳定性参数
    forget: 布尔就是否遭忘前向激活并在反向传播时重新计算
    test_vars: 用于测试的可选变量元组
    name: 可选的变量作用域名称

  Returns:
    形状 [batch, length, io_size] 的张量
  """
  io_size = x.get_shape().as_list()[-1]

  def forward_internal(x, f1, f2, scale, bias):
    """Forward function."""
    # split batch-wise to avoid exhausting memory in cast the batch is large
    # and the hidden layer is large.
    num_splits = 4
    x_flat = tf.reshape(x, [-1, 1, shape_list(x)[2]])
    xs = approximate_split(x_flat, num_splits)
    ys = []
    for i in range(num_splits):
      with tf.control_dependencies(ys[-1:]):
        n = layer_norm_compute(xs[i], epsilon, scale, bias)
        y = tf.nn.conv1d(n, f1, 1, "SAME")
        y = tf.nn.relu(y)
        y = tf.nn.conv1d(y, f2, 1, "SAME")
        ys.append(y)
    y = tf.concat(ys, 0)
    y = tf.reshape(y, shape_list(x))
    return y

  key = ("conv_hidden_relu_memory_efficient %s" % epsilon)
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:

    @function.Defun(compiled=True)
    def grad_fn(x, f1, f2, scale, bias, dy):
      """Gradient for efficiency."""
      with tf.control_dependencies([dy]):
        num_splits = 4
        x_shape = shape_list(x)
        flat_shape = [-1, 1, x_shape[2]]
        x = tf.reshape(x, flat_shape)
        dy = tf.reshape(dy, flat_shape)
        xs = approximate_split(x, num_splits)
        dys = approximate_split(dy, num_splits)
        dxs = []
        df1 = 0
        df2 = 0
        dscale = 0
        dbias = 0
        deps = []
        for i in range(num_splits):
          with tf.control_dependencies(deps):
            n = layer_norm_compute(xs[i], epsilon, scale, bias)
            y = tf.nn.conv1d(n, f1, 1, "SAME")
            y = tf.nn.relu(y)
            y = tf.nn.conv1d(y, f2, 1, "SAME")
            dxi, pdf1, pdf2, pdscale, pdbias = tf.gradients(
                ys=[y], xs=[xs[i], f1, f2, scale, bias], grad_ys=[dys[i]])
            df1 += pdf1
            df2 += pdf2
            dscale += pdscale
            dbias += pdbias
            dxs.append(dxi)
            deps = [dxi, df1, df2, dscale, dbias]
        with tf.control_dependencies(deps):
          dx = tf.concat(dxs, 0)
          dx = tf.reshape(dx, x_shape)
          return dx, df1, df2, dscale, dbias

    @function.Defun(
        grad_func=grad_fn, compiled=True, separate_compiled_gradients=True)
    def forward_fn(x, f1, f2, scale, bias):
      return forward_internal(x, f1, f2, scale, bias)

  with tf.variable_scope(name, default_name="ffn2", values=[x]):
    # TODO(noam): it would be nice to save memory by casting x to float16
    # here, but this causes problems with the gradients.  Figure out if there
    # is a way to leave the gradients as float32.
    if test_vars is not None:
      f1, f2, scale, bias = list(test_vars)
    else:
      f1 = tf.get_variable("f1", [1, io_size, filter_size])
      f2 = tf.get_variable("f2", [1, filter_size, io_size])
      scale, bias = layer_norm_vars(io_size)
    if forget:
      y = forward_fn(x, f1, f2, scale, bias)
    else:
      y = forward_internal(x, f1, f2, scale, bias)
    y.set_shape(x.get_shape())
    return y


def shape_list(x):
  """返回张量各维度大小列表，尽可能使用静态形状。
  
  这是 Tensor2Tensor 中最常用的工具函数之一。
  问题：tf.shape(x) 返回动态形状（运行时算），而 x.get_shape() 返回静态形状（可能有 None）。
  这个函数将两者结合：尽量使用静态值（更高效），如果有 None 则使用动态形状。
  
  Returns:
    列表，元素是静态整数（形状已知）或者 TF 张量（形状未知）
  """
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def list_product(els):
  """计算列表中所有元素的乘积（支持整数和 TF 张量）。
  
  Args:
    els: 整数或张量的列表
    
  Returns:
    所有元素的乘积
  """
  prod = els[0]
  for el in els[1:]:
    prod *= el
  return prod


def sample_with_temperature(logits, temperature, sampling_keep_top_k=-1):
  """从 logits 中采样：temperature=0 时为 argmax，>0 时为随机采样。

  temperature 控制采样的随机性：
  - temperature=0.0：確定性选择最高概率的 token（贪心解码）
  - temperature=1.0：按模型原始概率分布采样
  - temperature>1.0：更平滑的分布，增加多样性
  - temperature<1.0：更尖锐的分布，深化最常见的 token

  Args:
    logits: 张量
    temperature: 浮点数，0.0=argmax，1.0=随机采样
    sampling_keep_top_k: 如果不是 -1，只从 top-k 个 logits 中采样。
  Returns:
    比 logits 少一个维度的张量（采样结果）
  """
  if temperature == 0.0:
    # TF argmax doesn't handle >5 dimensions, so we reshape here.
    logits_shape = shape_list(logits)
    argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=1)
    return tf.reshape(argmax, logits_shape[:-1])
  else:
    tf.debugging.assert_greater(temperature, 0.0)

    if sampling_keep_top_k != -1:
      if sampling_keep_top_k <= 0:
        raise ValueError("sampling_keep_top_k must either be -1 or positive.")

      vocab_size = shape_list(logits)[1]

      k_largest = contrib.nn().nth_element(
          logits, n=sampling_keep_top_k, reverse=True)
      k_largest = tf.tile(tf.reshape(k_largest, [-1, 1]), [1, vocab_size])

      # Force every position that is not in the top k to have probability near
      # 0 by setting the logit to be very negative.
      logits = tf.where(tf.less_equal(logits, k_largest),
                        tf.ones_like(logits)*-1e6, logits)

    reshaped_logits = (
        tf.reshape(logits, [-1, shape_list(logits)[-1]]) / temperature)
    choices = tf.multinomial(reshaped_logits, 1)
    choices = tf.reshape(choices,
                         shape_list(logits)[:logits.get_shape().ndims - 1])
    return choices


def _select_top_k(logits, top_k):
  """将 logits 中非 top-k 的位置设为 -1e6（将其概率近似归零）。

  用于 top-k 采样：只允许从 logits 最高的 k 个类别中采样。
  支持按样本特制的 k 値（即每个样本可以使用不同的 k）。
  k=-1 时不进行任何过滤。

  Args:
    logits: 形状 [batch_size, ..., vocab_size] 的张量
    top_k: batch_size 大小的 k 向量

  Returns:
    与 logits 形状相同的张量，非 top-k 位置已设为 -1e6
  """
  vocab_size = logits.shape[-1]

  top_k = tf.where(
      tf.not_equal(top_k, -1), top_k,
      tf.ones_like(top_k) * vocab_size)

  return tf.where(
      tf.argsort(logits) < tf.reshape(top_k, [-1] + [1] *
                                      (len(logits.shape) - 1)), logits,
      tf.ones_like(logits) * -1e6)


def sample_temperature_per_example(logits, temperature, sampling_keep_top_k=-1):
  """每个样本使用独立采样温度进行随机采样。

  与 sample_with_temperature 类似，但允许批次内每个样本使用不同的温度。
  适用于需要批次内不同多样性的场景。

  Args:
    logits: 张量
    temperature: 浮点向量，尺寸与 logits 的批次大小一致
    sampling_keep_top_k: 如果不是 -1，只从 top-k 个 logits 中采样
  Returns:
    比 logits 少一个维度的张量（采样结果）
  """
  logits = _select_top_k(logits, sampling_keep_top_k)
  logits /= tf.reshape(temperature, [-1] + [1] * (len(logits.shape) - 1))
  reshaped_logits = tf.reshape(logits, [-1, shape_list(logits)[-1]])
  choices = tf.multinomial(reshaped_logits, 1)
  choices = tf.reshape(choices,
                       shape_list(logits)[:logits.get_shape().ndims - 1])
  return choices


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """创建只在指定对角带内为 1、其他为 0 的矩阵（任意形状）。
  
  用于创建注意力掩码（如因果掩码、近端注意力树、滑动窗口注意力）。
  
  num_lower=-1 表示对角线以下没有限制（连接所有之前的位置）。
  num_upper=-1 表示对角线以上没有限制。

  Args:
    rows: 输出矩阵的行数
    cols: 输出矩阵的列数
    num_lower: 下方对角带宽度（-1 表示不限制）
    num_upper: 上方对角带宽度（-1 表示不限制）
    out_shape: 用于重塑形状的可选形状

  Returns:
    大小为 rows * cols 重塑为 out_shape 的张量
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.linalg.band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band


def reshape_like_all_dims(a, b):
  """将 a reshape 为与 b 形状完全匹配。
  
  在非 eager mode 下试图传播静态形状。
  """
  ret = tf.reshape(a, tf.shape(b))
  if not tf.executing_eagerly():
    ret.set_shape(b.get_shape())
  return ret


def recompute_grad(fn):
  """装饰器：在反向传播时重新计算函数（梯度检置点 / Gradient Checkpointing）。

  这是一种用时间换内存的策略：
  - 前向传播：正常执行函数，但不保留中间激活（节约内存）
  - 反向传播：重新计算激活并计算梯度（耗时间）
  
  特别适用于大型模型训练，当激活擦叠内存是璶颈时。

  Args:
    fn: 以 Tensors 为位置参数、返回 Tensors 元组的函数

  Returns:
    封装后的 fn，调用时行为与 fn 相同，但其激活将被丢弃并在反向传播时重新计算
  """

  @functools.wraps(fn)
  def wrapped(*args):
    return _recompute_grad(fn, args)

  return wrapped


def _recompute_grad(fn, args):
  """See recompute_grad."""

  cached_vs = []
  cached_arg_scope = []

  def grad_fn(inputs, variables, outputs, output_grads):
    """Recompute outputs for gradient computation."""
    del outputs
    variables = [underlying_variable_ref(v) for v in variables]
    # Recompute outputs
    with tf.control_dependencies(output_grads):
      with contrib.framework().arg_scope(cached_arg_scope[0]):
        with tf.variable_scope(cached_vs[0], reuse=True):
          outputs = fn(*inputs)

    if not isinstance(outputs, (list, tuple)):
      outputs = [outputs]
    outputs = list(outputs)
    grads = tf.gradients(outputs, inputs + variables, output_grads)
    grad_inputs = grads[:len(inputs)]
    grad_vars = grads[len(inputs):]
    # TODO(rsepassi): Make fn_with_custom_grad work with bfloat16.
    # If the input gradients are bfloat16, it's assumed the variables are
    # bfloat16. This is a hack to ensure that grad_vars are the right type.
    if grad_inputs[0].dtype == tf.bfloat16:
      grad_vars = [tf.cast(grad_var, tf.bfloat16) for grad_var in grad_vars]
    return grad_inputs, grad_vars

  @fn_with_custom_grad(grad_fn)
  def fn_with_recompute(*args):
    cached_vs.append(tf.get_variable_scope())
    cached_arg_scope.append(contrib.framework().current_arg_scope())
    return fn(*args)

  return fn_with_recompute(*args)


def dense(x, units, **kwargs):
  """全连接层，功能与 tf.layers.dense 相同。
  
  对 layers().Dense(units) 的封装，额外支持 layer_collection 参数（用于 KFAC 优化器）。
  """
  layer_collection = kwargs.pop("layer_collection", None)
  activations = layers().Dense(units, **kwargs)(x)
  if layer_collection:
    # We need to find the layer parameters using scope name for the layer, so
    # check that the layer is named. Otherwise parameters for different layers
    # may get mixed up.
    layer_name = tf.get_variable_scope().name
    if (not layer_name) or ("name" not in kwargs):
      raise ValueError(
          "Variable scope and layer name cannot be empty. Actual: "
          "variable_scope={}, layer name={}".format(
              layer_name, kwargs.get("name", None)))

    layer_name += "/" + kwargs["name"]
    layer_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope=layer_name)
    assert layer_params
    if len(layer_params) == 1:
      layer_params = layer_params[0]

    tf.logging.info(
        "Registering dense layer to collection for tensor: {}".format(
            layer_params))

    x_shape = x.shape.as_list()
    if len(x_shape) == 3:
      # Handle [batch, time, depth] inputs by folding batch and time into
      # one dimension: reshaping inputs to [batchxtime, depth].
      x_2d = tf.reshape(x, [-1, x_shape[2]])
      activations_shape = activations.shape.as_list()
      activations_2d = tf.reshape(activations, [-1, activations_shape[2]])
      layer_collection.register_fully_connected_multi(
          layer_params, x_2d, activations_2d, num_uses=x_shape[1])
      activations = tf.reshape(activations_2d, activations_shape)
    else:
      layer_collection.register_fully_connected(layer_params, x, activations)
  return activations


def batch_dense(inputs,
                units,
                activation=None,
                kernel_initializer=None,
                reuse=None,
                name=None):
  """Multiply a batch of input matrices by a batch of parameter matrices.

  Each input matrix is multiplied by the corresponding parameter matrix.

  This is useful in a mixture-of-experts where the batch represents different
  experts with different inputs.

  Args:
    inputs: a Tensor with shape [batch, length, input_units]
    units: an integer
    activation: an optional activation function to apply to the output
    kernel_initializer: an optional initializer
    reuse: whether to reuse the varaible scope
    name: an optional string

  Returns:
    a Tensor with shape [batch, length, units]

  Raises:
    ValueError: if the "batch" or "input_units" dimensions of inputs are not
      statically known.
  """
  inputs_shape = shape_list(inputs)
  if len(inputs_shape) != 3:
    raise ValueError("inputs must have 3 dimensions")
  batch = inputs_shape[0]
  input_units = inputs_shape[2]
  if not isinstance(batch, int) or not isinstance(input_units, int):
    raise ValueError("inputs must have static dimensions 0 and 2")
  with tf.variable_scope(
      name,
      default_name="batch_dense",
      values=[inputs],
      reuse=reuse,
      dtype=inputs.dtype):
    if kernel_initializer is None:
      kernel_initializer = tf.random_normal_initializer(
          stddev=input_units**-0.5)
    w = tf.get_variable(
        "w", [batch, input_units, units],
        initializer=kernel_initializer,
        dtype=inputs.dtype)
    y = tf.matmul(inputs, w)
    if activation is not None:
      y = activation(y)
    return y


def mix(x1,
        x2,
        steps,
        is_training,
        min_prob=0.0,
        max_prob=1.0,
        mode="lin",
        simple=False,
        broadcast_last=False):
  """渐进式混合：从 x2 开始，随训练步数增加逐渐向 x1 过渡。
  
  用于实现计划采样（Scheduled Sampling）或课程学习（Curriculum Learning）：
  - 训练初期：主要使用 x2（如真实标签或简单输入）
  - 训练后期：主要使用 x1（如模型预测或困难输入）
  
  混合模式：
  - mode='lin'：使用线性衰减从 min_prob 到 max_prob 的混合概率
  - mode='exp'：使用指数衰减
  
  Args:
    x1: 目标张量（训练结束时主要使用的张量）
    x2: 起始张量（训练开始时主要使用的张量）
    steps: 完成混合过渡所需的训练步数
    is_training: 是否在训练模式（推理时直接返回 x1 或随机混合）
    min_prob: 混合中 x1 的最小概率
    max_prob: 混合中 x1 的最大概率（1.0 表示训练结束时完全使用 x1）
    mode: 概率增长模式，'lin'（线性）或 'exp'（指数）
    simple: 如果为 True，使用简单的元素级线性插值而非随机掩码
    broadcast_last: 如果为 True，在最后一维上广播混合 alpha
  
  Returns:
    混合后的张量
  """
  with tf.name_scope("mix"):
    if not is_training:
      if max_prob >= 1.0:
        return x1
      alpha_shape = shape_list(x1)
      if broadcast_last:
        alpha_shape = alpha_shape[:-1] + [1]
      alpha = tf.random_uniform(alpha_shape)
      alpha = to_float(tf.less(alpha, max_prob))
      return alpha * x1 + (1.0 - alpha) * x2

    def get_res():
      """Create the result.

      Separate function to speed it up later (see below).

      Returns:
        Tensor of mixed inputs.
      """
      if mode == "lin":
        alpha_p = inverse_lin_decay(steps)
      else:
        alpha_p = inverse_exp_decay(steps)
      alpha_p = alpha_p * (max_prob - min_prob) + min_prob
      if simple:
        return alpha_p * x1 + (1.0 - alpha_p) * x2
      alpha_shape = shape_list(x1)
      if broadcast_last:
        alpha_shape = alpha_shape[:-1] + [1]
      alpha = tf.random_uniform(alpha_shape)
      alpha = to_float(tf.less(alpha, alpha_p))
      return alpha * x1 + (1.0 - alpha) * x2

    if max_prob < 1.0:
      return get_res()

    # Prevent sampling after steps is passed to speed it up.
    if is_xla_compiled():
      return get_res()
    else:
      cur_step = tf.train.get_global_step()
      if cur_step is None:
        return x1  # Step not available, probably eval mode, don't mix.
      return tf.cond(tf.less(cur_step, steps), get_res, lambda: x1)


def brelu(x):
  """双极 ReLU（Bipolar ReLU），论文：https://arxiv.org/abs/1709.04054。
  
  将输入分为两半：
  - 前半部分应用 ReLU（只保留正值）
  - 后半部分应用 -ReLU(-x)（只保留负值，相当于倒置的 ReLU）
  
  这样的设计同时传递正值和负值信息，避免 ReLU 的单侧抑制。
  
  Args:
    x: 输入张量
    
  Returns:
    与 x 形状相同的张量
  """
  x_shape = shape_list(x)
  x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
  y1 = tf.nn.relu(x1)
  y2 = -tf.nn.relu(-x2)
  return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def belu(x):
  """双极 ELU（Bipolar ELU），论文：https://arxiv.org/abs/1709.04054。
  
  与 brelu 类似，但使用 ELU 代替 ReLU：
  - 前半部分应用 ELU（正值保留，负值平滑衰减到 -1）
  - 后半部分应用 -ELU(-x)
  
  ELU 相比 ReLU 在负区间有梯度，减少死神经元问题。
  
  Args:
    x: 输入张量
    
  Returns:
    与 x 形状相同的张量
  """
  x_shape = shape_list(x)
  x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
  y1 = tf.nn.elu(x1)
  y2 = -tf.nn.elu(-x2)
  return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def gelu(x):
  """高斯误差线性单元（Gaussian Error Linear Unit，GELU）激活函数。

  GELU 是 ReLU 的平滑版本，被 BERT、GPT 等大型语言模型广泛采用。
  原始论文：https://arxiv.org/abs/1606.08415
  
  数学公式：GELU(x) = x * Φ(x)
  其中 Φ(x) 是标准正态分布的累积分布函数（CDF），
  近似公式：x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
  
  与 ReLU 相比：
  - 不是硬截断（不会突然置零），而是平滑地接近零
  - 在负值区间也有小的梯度，不会产生死神经元
  - 实验上在 Transformer 中通常表现比 ReLU 更好

  Args:
    x: 浮点型张量

  Returns:
    应用 GELU 激活后的张量，形状与 x 相同
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def nac(x, depth, name=None, reuse=None):
  """神经累加器（Neural Accumulator，NAC），论文：https://arxiv.org/abs/1808.00508。
  
  NAC 设计用于处理数值运算（如加法、减法、计数）：
  - 权重矩阵 W = tanh(W_hat) * sigmoid(M_hat)，将参数约束在 [-1, 1]
  - 这种设计让模型学习精确的累加操作（整数加法等）
  - 比普通线性层更适合需要数值运算的任务
  
  Args:
    x: 输入张量
    depth: 输出维度
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    形状 [..., depth] 的输出张量
  """
  with tf.variable_scope(name, default_name="nac", values=[x], reuse=reuse):
    x_shape = shape_list(x)
    w = tf.get_variable("w", [x_shape[-1], depth])
    m = tf.get_variable("m", [x_shape[-1], depth])
    w = tf.tanh(w) * tf.nn.sigmoid(m)
    x_flat = tf.reshape(x, [-1, x_shape[-1]])
    res_flat = tf.matmul(x_flat, w)
    return tf.reshape(res_flat, x_shape[:-1] + [depth])


def nalu(x, depth, epsilon=1e-30, name=None, reuse=None):
  """神经算术逻辑单元（Neural Arithmetic Logic Unit，NALU），论文：https://arxiv.org/abs/1808.00508。
  
  NALU 扩展了 NAC，增加了乘法/除法运算能力：
  - 加法/减法路径：a = NAC(x)（在实数空间进行累加）
  - 乘法/除法路径：m = exp(NAC(log|x|))（在对数空间进行累加，等价于乘除法）
  - 门控组合：output = g * a + (1-g) * m
    其中 g = sigmoid(Dense(x)) 是学到的门控权重
  
  这样 NALU 可以同时学习加法和乘法等数值运算。
  
  Args:
    x: 输入张量
    depth: 输出维度
    epsilon: 对数运算的数值稳定性参数（避免 log(0)）
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    形状 [..., depth] 的输出张量
  """
  with tf.variable_scope(name, default_name="nalu", values=[x], reuse=reuse):
    x_shape = shape_list(x)
    x_flat = tf.reshape(x, [-1, x_shape[-1]])
    gw = tf.get_variable("w", [x_shape[-1], depth])
    g = tf.nn.sigmoid(tf.matmul(x_flat, gw))
    g = tf.reshape(g, x_shape[:-1] + [depth])
    a = nac(x, depth, name="nac_lin")
    log_x = tf.log(tf.abs(x) + epsilon)
    m = nac(log_x, depth, name="nac_log")
    return g * a + (1 - g) * tf.exp(m)


def argmax_with_score(logits, axis=None):
  """同时返回 argmax 索引和对应的最大值分数。
  
  比单独调用 tf.argmax 再做 gather 更高效，常用于解码时获取预测结果和置信度。
  
  Args:
    logits: 任意维度的张量，在最后一维或指定轴上取 argmax
    axis: 求 argmax 的轴，默认为最后一维
    
  Returns:
    (predictions, scores)：
    - predictions：argmax 索引，形状比 logits 少一维
    - scores：对应位置的最大值，形状与 predictions 相同
  """
  axis = axis or len(logits.get_shape()) - 1
  predictions = tf.argmax(logits, axis=axis)

  logits_shape = shape_list(logits)
  prefix_shape, vocab_size = logits_shape[:-1], logits_shape[-1]
  prefix_size = 1
  for d in prefix_shape:
    prefix_size *= d

  # Flatten to extract scores
  flat_logits = tf.reshape(logits, [prefix_size, vocab_size])
  flat_predictions = tf.reshape(predictions, [prefix_size])
  flat_indices = tf.stack(
      [tf.range(tf.to_int64(prefix_size)),
       tf.to_int64(flat_predictions)],
      axis=1)
  flat_scores = tf.gather_nd(flat_logits, flat_indices)

  # Unflatten
  scores = tf.reshape(flat_scores, prefix_shape)

  return predictions, scores


def log_prob_from_logits(logits, reduce_axis=-1):
  """从 logits 计算对数概率（log softmax）。
  
  log_prob = logits - log(sum(exp(logits)))
  使用 logsumexp 计算数值稳定，避免大值 exp 溢出。
  
  Args:
    logits: 任意形状的张量
    reduce_axis: 对哪个轴做归一化（默认最后一维）
    
  Returns:
    与 logits 形状相同的对数概率张量
  """
  return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)


def top_kth_iterative(x, k):
  """Compute the k-th top element of x on the last axis iteratively.

  This assumes values in x are non-negative, rescale if needed.
  It is often faster than tf.nn.top_k for small k, especially if k < 30.
  Note: this does not support back-propagation, it stops gradients!

  Args:
    x: a Tensor of non-negative numbers of type float.
    k: a python integer.

  Returns:
    a float tensor of the same shape as x but with 1 on the last axis
    that contains the k-th largest number in x.
  """
  # The iterative computation is as follows:
  #
  # cur_x = x
  # for _ in range(k):
  #   top_x = maximum of elements of cur_x on the last axis
  #   cur_x = cur_x where cur_x < top_x and 0 everywhere else (top elements)
  #
  # We encode this computation in a TF graph using tf.foldl, so the inner
  # part of the above loop is called "next_x" and tf.foldl does the loop.
  def next_x(cur_x, _):
    top_x = tf.reduce_max(cur_x, axis=-1, keep_dims=True)
    return cur_x * to_float(cur_x < top_x)
  # We only do k-1 steps of the loop and compute the final max separately.
  fin_x = tf.foldl(next_x, tf.range(k - 1), initializer=tf.stop_gradient(x),
                   parallel_iterations=2, back_prop=False)
  return tf.stop_gradient(tf.reduce_max(fin_x, axis=-1, keep_dims=True))


def top_1_tpu(inputs):
  """find max and argmax over the last dimension.

  Works well on TPU

  Args:
    inputs: A tensor with shape [..., depth]

  Returns:
    values: a Tensor with shape [...]
    indices: a Tensor with shape [...]
  """
  inputs_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
  mask = tf.to_int32(tf.equal(inputs_max, inputs))
  index = tf.range(tf.shape(inputs)[-1]) * mask
  return tf.squeeze(inputs_max, -1), tf.reduce_max(index, axis=-1)


def index_last_dim_with_indices(x, indices):
  """Use indices to index into the last axis of x.

  This can be useful for recovering the actual probabilities of a sample from a
  probability distribution.

  Args:
    x: Tensor, n-d.
    indices: Tensor, (n-1)-d, where the dimension sizes match the first (n-1)
      dimensions of x. The values of indices will be used to index into the last
      axis of x.

  Returns:
    Tensor, (n-1)-d.
  """
  assert len(x.shape) == len(indices.shape) + 1

  x_shape = shape_list(x)
  vocab_size = x_shape[-1]

  flat_x = tf.reshape(x, [list_product(x_shape[:-1]), vocab_size])
  flat_indices = tf.reshape(indices, [list_product(x_shape[:-1])])

  idx = tf.stack(
      [
          tf.range(tf.to_int64(shape_list(flat_indices)[0])),
          tf.to_int64(flat_indices)
      ],
      axis=1)
  flat_x_idx = tf.gather_nd(flat_x, idx)

  x_idx = tf.reshape(flat_x_idx, x_shape[:-1])

  return x_idx


def should_generate_summaries():
  """判断当前上下文是否适合生成 TensorBoard 摘要（Summaries）。
  
  有两种情况下不应生成摘要：
  1. 在 tf.while_loop() 内部：摘要在循环中效果不好（会产生大量重复数据）
  2. 在 variable_scope reuse=True 的情况下：避免为不同数据分片生成单独的摘要

  Returns:
    bool: True 表示应该生成摘要，False 表示跳过摘要生成
  """
  name_scope = contrib.framework().get_name_scope()
  if name_scope and "while/" in name_scope:
    # Summaries don't work well within tf.while_loop()
    return False
  if tf.get_variable_scope().reuse:
    # Avoid generating separate summaries for different data shards
    return False
  return True


def reshape_like(a, b):
  """将 a reshape 成与 b 除最后一维外形状相同（最后一维保留 a 的维度）。
  
  常用于在保持最后一维（feature 维度）不变的情况下，调整前置维度。
  
  Args:
    a: 需要 reshape 的张量
    b: 提供目标形状（除最后一维）的参考张量
    
  Returns:
    reshape 后的 a：前置维度与 b 匹配，最后一维与 a 的原始最后一维相同
  """
  ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
  if not tf.executing_eagerly():
    ret.set_shape(b.get_shape().as_list()[:-1] + a.get_shape().as_list()[-1:])
  return ret


def summarize_video(video, prefix, max_outputs=1):
  """Summarize the video using image summaries starting with prefix."""
  video_shape = shape_list(video)
  if len(video_shape) != 5:
    raise ValueError("Assuming videos given as tensors in the format "
                     "[batch, time, height, width, channels] but got one "
                     "of shape: %s" % str(video_shape))
  if tf.executing_eagerly():
    return
  if video.get_shape().as_list()[1] is None:
    tf.summary.image(
        "%s_last_frame" % prefix,
        tf.cast(video[:, -1, :, :, :], tf.uint8),
        max_outputs=max_outputs)
  else:
    for k in range(video_shape[1]):
      tf.summary.image(
          "%s_frame_%d" % (prefix, k),
          tf.cast(video[:, k, :, :, :], tf.uint8),
          max_outputs=max_outputs)


def cast_like(x, y):
  """将 x 的数据类型转换为与 y 相同（如果类型不同则进行转换）。
  
  在混合精度训练中常用，确保两个张量的数据类型一致。
  如果已经相同则直接返回 x，避免不必要的转换操作。
  
  Args:
    x: 需要转换类型的张量
    y: 提供目标数据类型的参考张量
    
  Returns:
    与 x 值相同但数据类型与 y 一致的张量
  """
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x


def make_even_size(x):
  """将张量在第 1 和第 2 维度填充到偶数大小（仅在必要时填充）。
  
  某些卷积操作（如步幅为 2 的卷积）要求输入的空间维度为偶数。
  此函数通过在末尾填充零来确保这一点，同时尽量不破坏静态形状。
  
  Args:
    x: 至少 3 维的张量（如 [batch, height, width, channels]）
    
  Returns:
    第 1 和第 2 维度被填充到偶数大小的张量
    
  Raises:
    AssertionError: 如果张量维度不足 3
  """
  x_shape = x.get_shape().as_list()
  assert len(x_shape) > 2, "Only 3+-dimensional tensors supported."
  shape = [dim if dim is not None else -1 for dim in x_shape]
  new_shape = x_shape  # To make sure constant shapes remain constant.
  if x_shape[1] is not None:
    new_shape[1] = 2 * int(math.ceil(x_shape[1] * 0.5))
  if x_shape[2] is not None:
    new_shape[2] = 2 * int(math.ceil(x_shape[2] * 0.5))
  if shape[1] % 2 == 0 and shape[2] % 2 == 0:
    return x
  if shape[1] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
    x.set_shape(new_shape)
    return x
  if shape[2] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
    x.set_shape(new_shape)
    return x
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
  x.set_shape(new_shape)
  return x


def sliced_gan_loss(input1,
                    input2,
                    discriminator,
                    num_vecs,
                    do_random_vecs=True,
                    do_tanh=True,
                    return_logits=False):
  """Loss inspired by the sliced WGAN paper: https://arxiv.org/abs/1804.01947.

  Puts input1 and input2 through the provided discriminator to get logits.
  Then, computes num_vecs random projections of the logits, sorts them on
  the batch dimension and returns the L2 loss between the sorted vectors.
  See the above-mentioned paper for the reasoning behind it.

  Args:
    input1: first discriminator inputs.
    input2: second discriminator inputs.
    discriminator: inputs -> logits function.
    num_vecs: how many random vectors to use for projections.
    do_random_vecs: whether to use random vectors or just tanh of the logits.
    do_tanh: if true (default) we'll also just use tanh of the logits.
    return_logits: Whether or not to return the logits.

  Returns:
    The generator loss, i.e., the sliced approximation of the distance between
    the projected distributions (warning: discriminator should maximize it).
  """
  with tf.variable_scope("sliced_gan"):
    with tf.variable_scope("discriminator"):
      logits1 = discriminator(input1)
    with tf.variable_scope("discriminator", reuse=True):
      logits2 = discriminator(input2)

    if do_random_vecs:
      random_vecs = tf.nn.l2_normalize(
          tf.random_uniform([shape_list(logits1)[-1], num_vecs]), axis=0)

    def get_sorted_projections(x):
      """Make projections of x and sort them on the batch dimension."""
      x = tf.reshape(x, [-1, shape_list(x)[-1]])
      batch_size = shape_list(x)[0]
      if do_random_vecs and do_tanh:
        n = tf.nn.l2_normalize(x, axis=1)
        proj = tf.concat([tf.matmul(n, random_vecs), tf.tanh(n)], axis=1)
      elif do_random_vecs:
        n = tf.nn.l2_normalize(x, axis=1)
        proj = tf.matmul(n, random_vecs)
      else:
        proj = tf.tanh(x)
      proj = tf.transpose(proj, [1, 0])  # [num_vecs, batch] after this.

      if is_xla_compiled():
        proj_dtype = proj.dtype
        proj = tf.cast(proj, tf.bfloat16)

        # Currently TPU only supports 1-D top_k calls.
        map_fn = lambda x: tf.nn.top_k(x, k=batch_size, sorted=True)[0]
        values = tf.map_fn(map_fn, proj)

        values = tf.cast(values, proj_dtype)
      else:
        values, _ = tf.nn.top_k(proj, k=batch_size, sorted=True)

      return values

    proj1 = get_sorted_projections(logits1)
    proj2 = get_sorted_projections(logits2)
    dist = tf.reduce_mean(tf.squared_difference(proj1, proj2))
    if return_logits:
      return dist, logits1, logits2
    return dist


def lrelu(input_, leak=0.2, name="lrelu"):
  """Leaky ReLU（带泄漏的修正线性单元）激活函数。
  
  与标准 ReLU 不同，Leaky ReLU 在 x<0 时输出 leak * x 而不是 0：
  - x >= 0：输出 x
  - x < 0：输出 leak * x（保持小的负梯度，避免死神经元）
  
  常用于 GAN 的判别器中，为负值输入保留梯度信息。
  
  Args:
    input_: 输入张量
    leak: 负值区间的斜率（默认 0.2）
    name: 操作名称
    
  Returns:
    与输入形状相同的张量
  """
  return tf.maximum(input_, leak * input_, name=name)


def deep_discriminator(x,
                       batch_norm,
                       is_training,
                       filters=64,
                       filter_size=4,
                       stride=2,
                       output_size=1024):
  """基于 InfoGAN 结构的深度判别器（GAN 中用于区分真实和生成样本）。
  
  结构：
  - Conv2D(filters, filter_size, stride=2) + Leaky ReLU  # 下采样
  - Conv2D(2*filters, filter_size, stride=2) + BatchNorm + Leaky ReLU  # 进一步下采样
  - Flatten -> Dense(output_size) + BatchNorm + Leaky ReLU  # 全连接
  
  Args:
    x: 输入图像张量，形状 [batch, height, width, channels]
    batch_norm: 是否使用批归一化
    is_training: 是否在训练模式
    filters: 基础卷积通道数
    filter_size: 卷积核大小
    stride: 卷积步幅（控制下采样率）
    output_size: 最后全连接层的输出大小
    
  Returns:
    判别器特征向量，形状 [batch, output_size]
  """
  with tf.variable_scope(
      "discriminator", initializer=tf.random_normal_initializer(stddev=0.02)):
    batch_size, height, width = shape_list(x)[:3]  # pylint: disable=unbalanced-tuple-unpacking
    net = layers().Conv2D(
        filters, filter_size, strides=stride, padding="SAME", name="conv1")(x)
    net = lrelu(net)
    net = layers().Conv2D(
        2 * filters,
        filter_size,
        strides=stride,
        padding="SAME",
        name="conv2")(net)
    # [bs, h/4, w/4, 128]
    if batch_norm:
      net = layers().BatchNormalization(
          training=is_training, momentum=0.999, name="d_bn2")(net)
    net = lrelu(net)
    size = height * width
    x_shape = x.get_shape().as_list()
    if x_shape[1] is None or x_shape[2] is None:
      net = tf.reduce_mean(net, axis=[1, 2])
    else:
      net = tf.reshape(net, [batch_size, size * 8])
    net = layers().Dense(output_size, name="d_fc3")(net)
    if batch_norm:
      net = layers().BatchNormalization(
          training=is_training, momentum=0.999, name="d_bn3")(net)
    net = lrelu(net)
    return net


def instance_norm(x):
  """实例归一化层（Instance Normalization）。
  
  与批归一化（Batch Norm）不同，实例归一化在每个样本的每个通道上独立计算均值和方差：
  - 批归一化：在 batch 和空间维度上归一化（归一化跨样本统计）
  - 实例归一化：只在空间维度（H, W）上归一化（每个样本独立）
  
  实例归一化在风格迁移（Style Transfer）和 CycleGAN 等任务中常用，
  因为它可以将每个样本的风格信息归一化掉，只保留内容。
  
  Args:
    x: 输入张量，形状 [batch, height, width, channels]
    
  Returns:
    实例归一化后的张量，形状与 x 相同
  """
  with tf.variable_scope("instance_norm"):
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    scale = tf.get_variable(
        "scale", [x.get_shape()[-1]],
        initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
    offset = tf.get_variable(
        "offset", [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
    out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

    return out


def general_conv(x,
                 num_filters=64,
                 filter_size=7,
                 stride=1,
                 stddev=0.02,
                 padding="VALID",
                 name="conv",
                 do_norm="instance",
                 do_relu=True,
                 relufactor=0):
  """通用卷积层：卷积 + 可选归一化 + 可选激活（用于 CycleGAN 等图像生成网络）。
  
  这是一个用于图像生成网络的通用卷积块，封装了三个可选操作：
  1. 卷积：标准 Conv2D，可指定卷积核大小、步幅和填充方式
  2. 归一化：支持 'layer'（层归一化）、'instance'（实例归一化）或不归一化
  3. 激活：支持 ReLU、Leaky ReLU 或不激活
  
  Args:
    x: 输入张量
    num_filters: 卷积输出通道数
    filter_size: 卷积核大小
    stride: 卷积步幅
    stddev: 卷积核初始化标准差
    padding: 填充方式
    name: 变量作用域名称
    do_norm: 归一化类型，'layer'（层归一化）、'instance'（实例归一化）或 False（不归一化）
    do_relu: 是否应用激活函数
    relufactor: Leaky ReLU 的负斜率（0 表示使用标准 ReLU）
    
  Returns:
    卷积块输出张量
  """
  with tf.variable_scope(name):
    x = layers().Conv2D(
        num_filters,
        filter_size,
        stride,
        padding,
        activation=None,
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        bias_initializer=tf.constant_initializer(0.0))(x)
    if do_norm == "layer":
      x = layer_norm(x)
    elif do_norm == "instance":
      x = instance_norm(x)

    if do_relu:
      if relufactor == 0:
        x = tf.nn.relu(x, "relu")
      else:
        x = lrelu(x, leak=relufactor)

    return x


def patch_discriminator(x, filters=64, filter_size=5, n=4,
                        name="patch_discrim"):
  """PatchGAN 判别器（对图像的随机补丁进行判别）。
  
  PatchGAN 不判断整张图像是真实还是生成的，而是在随机裁剪的图像补丁上进行判别。
  这样可以捕获局部纹理细节，比全局判别器更适合图像翻译任务（如 CycleGAN）。
  
  结构：
  - 先将输入随机裁剪到 1/4 大小（height/4, width/4）
  - 然后应用 n 层 general_conv（逐层增加通道数：filters, 2*filters, 4*filters, ...）
  - 最后在空间维度上取均值
  
  Args:
    x: 输入张量，形状 [batch, height, width, channels]
    filters: 基础卷积通道数
    filter_size: 卷积核大小
    n: 卷积层数
    name: 变量作用域名称
    
  Returns:
    形状 [batch] 的判别分数
  """
  with tf.variable_scope(name):
    x_shape = shape_list(x)
    spatial_dims = [x_shape[1] // 4, x_shape[2] // 4]
    x = tf.random_crop(x, [x_shape[0]] + spatial_dims + [x_shape[3]])
    for i in range(n):
      x = general_conv(
          x=x,
          num_filters=filters * 2**i,
          filter_size=filter_size,
          stride=2 if i != n - 1 else 1,
          stddev=0.02,
          padding="SAME",
          name="c%d" % i,
          do_norm="instance" if i != 0 else False,
          do_relu=i != n - 1,
          relufactor=0.2)
    x = tf.reduce_mean(x, [1, 2])
    return x


def mean_with_attention(x, name, num_heads=4):
  """使用均值和注意力加权求和来压缩空间维度。
  
  同时计算：
  1. 全局均值池化（global mean pooling）
  2. 多头注意力加权求和（weighted sum with learned attention weights）
  然后将两者拼接后通过线性层输出。
  
  这比单纯的全局均值池化更好，因为注意力机制可以学习对重要区域赋予更高权重。
  
  Args:
    x: 输入张量，形状 [batch, height, width, channels]
    name: 变量作用域名称
    num_heads: 注意力头数
    
  Returns:
    形状 [batch, 2*channels] 的输出张量（均值特征 + 注意力特征的拼接）
  """
  with tf.variable_scope(name):
    shape = shape_list(x)
    m = tf.reduce_mean(x, [1, 2])
    a = layers().Dense(num_heads, name="mean_attn")(x)
    s = tf.reshape(a, [shape[0], -1, num_heads])
    s = tf.nn.softmax(s, axis=1)
    s = tf.reshape(s, shape[:-1] + [1, num_heads])
    am = tf.reduce_mean(tf.expand_dims(x, axis=-1) * s, [1, 2])
    l = tf.concat([am, tf.expand_dims(m, axis=-1)], axis=-1)
    return layers().Dense(2 * shape[-1], name="mean_attn_final")(
        tf.reshape(l, [shape[0], (num_heads+1) * shape[-1]]))


def single_discriminator(x, filters=128, kernel_size=8,
                         strides=4, pure_mean=False):
  """简单的单层卷积判别器（GAN 中区分真实和生成样本）。
  
  结构：
  - Conv2D(filters, kernel_size, strides)：提取特征
  - 空间压缩：全局均值或 mean_with_attention
  
  Args:
    x: 输入张量，形状 [batch, height, width, channels]
    filters: 卷积通道数
    kernel_size: 卷积核大小
    strides: 卷积步幅（控制下采样率）
    pure_mean: True 时使用简单均值池化，False 时使用注意力加权均值
    
  Returns:
    形状 [batch, ?] 的判别特征向量
  """
  with tf.variable_scope("discriminator"):
    net = layers().Conv2D(
        filters, kernel_size, strides=strides, padding="SAME", name="conv1")(x)
    if pure_mean:
      net = tf.reduce_mean(net, [1, 2])
    else:
      net = mean_with_attention(net, "mean_with_attention")
    return net


def double_discriminator(x, filters1=128, filters2=None,
                         kernel_size=8, strides=4, pure_mean=False):
  """两层卷积判别器，拼接两层的特征输出。
  
  比 single_discriminator 提取更丰富的特征：
  - 第一层提取低级特征，通过空间压缩得到 net1
  - 第二层（在第一层基础上）提取高级特征，通过空间压缩得到 net2
  - 拼接 net1 和 net2 输出，提供多尺度的特征表示
  
  Args:
    x: 输入张量，形状 [batch, height, width, channels]
    filters1: 第一层卷积通道数
    filters2: 第二层卷积通道数（默认为 4*filters1）
    kernel_size: 卷积核大小
    strides: 卷积步幅（控制下采样率）
    pure_mean: True 时使用简单均值池化，False 时使用注意力加权均值
    
  Returns:
    形状 [batch, ?] 的拼接判别特征向量（包含两层特征）
  """
  if filters2 is None:
    filters2 = 4 * filters1
  with tf.variable_scope("discriminator"):
    batch_size = shape_list(x)[0]
    net = layers().Conv2D(
        filters1, kernel_size, strides=strides, padding="SAME", name="conv1")(x)
    if pure_mean:
      net1 = tf.reduce_mean(net, [1, 2])
    else:
      net1 = mean_with_attention(net, "mean_with_attention1")
      tf.reshape(net, [batch_size, -1])
    net = tf.nn.relu(net)
    net = layers().Conv2D(
        filters2, kernel_size, strides=strides, padding="SAME",
        name="conv2")(net)
    if pure_mean:
      net2 = tf.reduce_mean(net, [1, 2])
    else:
      net2 = mean_with_attention(net, "mean_with_attention2")
    return tf.concat([net1, net2], axis=-1)


def upscale(inputs, f, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
  """将图像在高度和宽度维度上放大 f 倍。
  
  Args:
    inputs: 输入张量，形状 [batch, height, width, channels]
    f: 放大倍数（整数）
    method: 插值方式（默认最近邻插值，也可使用双线性插值等）
    
  Returns:
    放大后的张量，形状 [batch, height*f, width*f, channels]
  """
  height, width = shape_list(inputs)[1:3]  # pylint: disable=unbalanced-tuple-unpacking
  return tf.image.resize_images(inputs, (height * f, width * f), method)


def tpu_safe_image_summary(image):
  """将图像转换为适合在 TensorBoard 摘要中显示的格式（TPU 兼容版本）。
  
  TPU 上的 TensorBoard 摘要只支持 float32 类型的图像，
  而 CPU/GPU 上的 tf.summary.image 通常期望 uint8 类型。
  此函数根据环境选择合适的格式。
  
  Args:
    image: 图像张量
    
  Returns:
    转换后的图像张量（TPU 上为 float32，否则为 uint8）
  """
  if is_xla_compiled():
    # We only support float32 images at the moment due to casting complications.
    if image.dtype != tf.float32:
      image = to_float(image)
  else:
    image = tf.cast(image, tf.uint8)
  return image


# This has been (shamefully) copied from
# GitHub tensorflow/models/blob/master/research/slim/nets/cyclegan.py
#
# tensorflow/models cannot be pip installed, and even if it were we don't want
# to depend on all the models in it.
#
# Therefore copying and forgoing any more bugfixes into it is the most
# expedient way to use this function.
def cyclegan_upsample(net, num_outputs, stride, method="conv2d_transpose"):
  """Upsamples the given inputs.

  Args:
    net: A Tensor of size [batch_size, height, width, filters].
    num_outputs: The number of output filters.
    stride: A list of 2 scalars or a 1x2 Tensor indicating the scale,
      relative to the inputs, of the output dimensions. For example, if kernel
      size is [2, 3], then the output height and width will be twice and three
      times the input size.
    method: The upsampling method: 'nn_upsample_conv',
      'bilinear_upsample_conv', or 'conv2d_transpose'.

  Returns:
    A Tensor which was upsampled using the specified method.

  Raises:
    ValueError: if `method` is not recognized.
  """

  with tf.variable_scope("upconv"):
    net_shape = tf.shape(net)
    height = net_shape[1]
    width = net_shape[2]

    # Reflection pad by 1 in spatial dimensions (axes 1, 2 = h, w) to make a
    # 3x3 "valid" convolution produce an output with the same dimension as the
    # input.
    spatial_pad_1 = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])

    if method == "nn_upsample_conv":
      net = tf.image.resize_nearest_neighbor(
          net, [stride[0] * height, stride[1] * width])
      net = tf.pad(net, spatial_pad_1, "REFLECT")
      net = layers().Conv2D(
          num_outputs, (3, 3), activation=tf.nn.relu)(net)
    elif method == "bilinear_upsample_conv":
      net = tf.image.resize_bilinear(net,
                                     [stride[0] * height, stride[1] * width])
      net = tf.pad(net, spatial_pad_1, "REFLECT")
      net = layers().Conv2D(
          num_outputs, (3, 3), activation=tf.nn.relu)(net)
    elif method == "conv2d_transpose":
      # This corrects 1 pixel offset for images with even width and height.
      # conv2d is left aligned and conv2d_transpose is right aligned for even
      # sized images (while doing "SAME" padding).
      # Note: This doesn"t reflect actual model in paper.
      net = layers().Conv2DTranspose(
          num_outputs, (3, 3), strides=stride, activation=tf.nn.relu)(net)
      net = net[:, 1:, 1:, :]
    else:
      raise ValueError("Unknown method: [%s]" % method)

    return net


def weight_targeting(w, k):
  """权重级别的幅度剪枝（Weight-level Magnitude Pruning）。
  
  对于每个输出单元，找出幅度最小的 k 个权重连接，并返回一个掩码：
  - 掩码为 0：对应的连接是目标（最小的 k 个），可能被剪枝
  - 掩码为 1：对应的连接不是目标
  
  与 unit_targeting 的区别：
  - weight_targeting：逐个权重排序（更细粒度）
  - unit_targeting：逐个输出单元排序（以整个神经元为单位）
  
  Args:
    w: 权重张量，最后一维为输出维度
    k: 每个输出单元中目标的权重数量
    
  Returns:
    与 w 形状相同的浮点掩码张量（1.0 表示非目标，0.0 表示目标）
  """
  k = tf.to_int32(k)
  w_shape = shape_list(w)
  size = tf.to_int32(tf.reduce_prod(w_shape[:-1]))
  w = tf.reshape(w, [size, w_shape[-1]])

  transpose_w = tf.transpose(w)
  thres = contrib.framework().sort(tf.abs(transpose_w), axis=1)[:, k]
  mask = to_float(thres[None, :] >= tf.abs(w))

  return tf.reshape(mask, w_shape)


def unit_targeting(w, k):
  """单元级别的幅度剪枝（Unit-level Magnitude Pruning）。
  
  对所有输出单元按 L2 范数排序，找出范数最小的 k 个单元，并返回掩码：
  - 掩码为 0：对应的单元是目标（最小的 k 个），可能被剪枝
  - 掩码为 1：对应的单元不是目标
  
  与 weight_targeting 相比，这里以整个输出神经元为单位进行剪枝（结构化剪枝）。
  
  Args:
    w: 权重张量，最后一维为输出维度
    k: 目标（剪枝）的输出单元数量
    
  Returns:
    与 w 形状相同的浮点掩码张量（1.0 表示非目标，0.0 表示目标）
  """
  k = tf.to_int32(k)
  w_shape = shape_list(w)
  size = tf.to_int32(tf.reduce_prod(w_shape[:-1]))
  w = tf.reshape(w, [size, w_shape[-1]])

  norm = tf.norm(w, axis=0)
  thres = contrib.framework().sort(norm, axis=0)[k]
  mask = to_float(thres >= norm)[None, :]
  mask = tf.tile(mask, [size, 1])

  return tf.reshape(mask, w_shape)


def td_conv(inputs,
            filters,
            kernel_size,
            targeting_count,
            targeting_fn,
            keep_prob,
            is_training,
            do_prune=True,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            name=None,
            reuse=None):
  """带目标 Dropout 的卷积层（Targeted Dropout Convolution）。
  
  目标 Dropout 对权重中最小幅度的部分应用 Dropout，
  训练过程中自动识别并压制不重要的权重，使剪枝后的模型性能损失更小。
  
  参考论文：'Targeted Dropout for Posthoc Pruning'
  Aidan N. Gomez, Ivan Zhang, Kevin Swersky, Yarin Gal, and Geoffrey E. Hinton.
  
  Args:
    inputs: 输入张量
    filters: 卷积输出通道数
    kernel_size: 卷积核大小（整数）
    targeting_count: 每次 dropout 目标的权重数量（传给 targeting_fn）
    targeting_fn: 选择目标权重的函数，格式 fn(weights, k) -> bool mask
      返回 True 表示该权重被目标（将被 dropout）
    keep_prob: 被目标权重的保留概率（1.0 表示不丢弃）
    is_training: 是否在训练模式
    do_prune: 是否在推理时进行硬剪枝（将目标权重永久置零）
    strides: 卷积步幅
    padding: 填充方式
    data_format: 数据格式，'channels_last'（NHWC）或 'channels_first'（NCHW）
    dilation_rate: 膨胀率
    activation: 可选激活函数
    use_bias: 是否使用偏置
    kernel_initializer: 卷积核初始化器
    bias_initializer: 偏置初始化器
    name: 变量作用域名称
    reuse: 是否重用变量
    
  Returns:
    卷积输出张量
  """
  with tf.variable_scope(name, default_name="td_conv", reuse=reuse):
    nhwc = data_format == "channels_last"
    in_dim = shape_list(inputs)[-1] if nhwc else shape_list(inputs)[1]

    kernel_shape = [kernel_size, kernel_size, in_dim, filters]
    w = tf.get_variable(
        "DW", shape=kernel_shape, initializer=kernel_initializer)
    if use_bias:
      b = tf.get_variable("b", shape=[filters], initializer=bias_initializer)

    if keep_prob < 1.0:
      w = targeted_dropout(
          w,
          targeting_count,
          keep_prob,
          targeting_fn,
          is_training,
          do_prune=do_prune)

    if isinstance(strides, int):
      strides = [strides, strides]
    if isinstance(dilation_rate, int):
      dilation_rate = [dilation_rate, dilation_rate]

    if nhwc:
      strides = [1, strides[0], strides[1], 1]
      dilation_rate = [1, dilation_rate[0], dilation_rate[1], 1]
    else:
      strides = [1, 1, strides[0], strides[1]]
      dilation_rate = [1, 1, dilation_rate[0], dilation_rate[1]]

    y = tf.nn.conv2d(
        inputs,
        w,
        strides,
        padding,
        data_format="NHWC" if nhwc else "NCHW",
        dilations=dilation_rate,
        name=None)

    if use_bias:
      y += b

    if activation:
      y = activation(y)

    return y


def targeted_dropout(inputs,
                     k,
                     keep_prob,
                     targeting_fn,
                     is_training,
                     do_prune=False):
  """Applies targeted dropout.

  Applies dropout at a rate of `1 - keep_prob` to only those elements of
  `inputs` marked by `targeting_fn`. See below and paper for more detail:

  "Targeted Dropout for Posthoc Pruning" Aidan N. Gomez, Ivan Zhang,
    Kevin Swersky, Yarin Gal, and Geoffrey E. Hinton.

  Args:
    inputs: Tensor, inputs to apply targeted dropout to.
    k: Scalar Tensor or python scalar, sets the number of elements to target in
      `inputs`. Must be within `[0, tf.shape(x)[-1]]` and compatible with
      second argument of `targeting_fn`.
    keep_prob: Scalar Tensor, passed as `tf.nn.dropout`'s `keep_prob` argument.
    targeting_fn: callable `fn(inputs, k) -> Boolean Tensor`, produces a
      boolean mask the same shape as `inputs` where True indicates an element
      will be dropped, and False not.
    is_training: bool, indicates whether currently training.
    do_prune: bool, indicates whether to prune the `k * (1 - keep_prob)`
      elements of `inputs` expected to be dropped each forwards pass.

  Returns:
    Tensor, same shape and dtype as `inputs`.
  """
  if not is_training and do_prune:
    k = tf.round(to_float(k) * to_float(1. - keep_prob))

  mask = targeting_fn(inputs, k)
  mask = tf.cast(mask, inputs.dtype)

  if is_training:
    return inputs * (1 - mask) + tf.nn.dropout(inputs, keep_prob) * mask
  elif do_prune:
    return inputs * (1 - mask)
  else:
    return inputs


def kl_divergence(mu, log_var, mu_p=0.0, log_var_p=0.0):
  """计算对角高斯分布 N(mu, exp(log_var)) 相对于先验分布的 KL 散度。
  
  KL 散度（Kullback-Leibler Divergence）用于变分自编码器（VAE）的正则化损失：
  - 后验分布：N(mu, exp(log_var))
  - 先验分布：N(mu_p, exp(log_var_p))，默认为标准正态 N(0, 1)
  
  KL(q||p) = 期望 q 对 log(q/p) 的积分，对角高斯分布有解析解。
  
  这个损失促使编码器输出接近先验分布，实现了 VAE 的潜在空间正则化，
  使潜在空间连续且有意义，可以从中采样生成新样本。

  Args:
    mu: 后验分布的均值参数，形状 [batch, latent_dim]
    log_var: 后验分布的对数方差，形状 [batch, latent_dim]
    mu_p: 先验分布的均值（默认 0.0）
    log_var_p: 先验分布的对数方差（默认 0.0，即方差为 1）
    
  Returns:
    标量 KL 散度损失（对 batch 取均值后对所有维度求和）
  """

  batch_size = shape_list(mu)[0]
  prior_distribution = tfp.distributions.Normal(
      mu_p, tf.exp(tf.multiply(0.5, log_var_p)))
  posterior_distribution = tfp.distributions.Normal(
      mu, tf.exp(tf.multiply(0.5, log_var)))
  kld = tfp.distributions.kl_divergence(posterior_distribution,
                                        prior_distribution)
  return tf.reduce_sum(kld) / to_float(batch_size)


def sparse_equals_constant(constant, tensor):
  """判断稀疏张量的每个非零值是否等于指定常量。
  
  Args:
    constant: 比较的目标常量值
    tensor: 稀疏张量（SparseTensor）
    
  Returns:
    与 tensor 结构相同的稀疏布尔张量（values 为比较结果）
  """
  return tf.SparseTensor(
      indices=tensor.indices,
      dense_shape=tensor.dense_shape,
      values=tf.equal(tensor.values, constant))


def sparse_expand_dims(tensor, current_num_dims, axis=0):
  """在稀疏张量的指定位置插入大小为 1 的新维度。
  
  等价于稀疏张量版本的 tf.expand_dims。
  
  Args:
    tensor: 稀疏张量（SparseTensor）
    current_num_dims: 当前张量的维度数
    axis: 插入新维度的位置（0 表示最前面，-1 表示最后面）
    
  Returns:
    在指定位置插入新维度后的稀疏张量
  """
  if axis == -1:
    axis = current_num_dims

  new_col = tf.zeros([tf.shape(tensor.indices)[0]], dtype=tf.int64)
  cols = tf.unstack(tensor.indices, axis=1, num=current_num_dims)
  shape = tf.unstack(tensor.dense_shape, num=current_num_dims)
  new_indices = tf.stack(cols[:axis] + [new_col] + cols[axis:], axis=1)
  return tf.SparseTensor(
      indices=new_indices,
      values=tensor.values,
      dense_shape=tf.stack(shape[:axis] + [1] + shape[axis:]))


def sparse_add_constant(constant, tensor):
  """将常量加到稀疏张量的每个非零值上。
  
  Args:
    constant: 要加的常量值
    tensor: 稀疏张量（SparseTensor）
    
  Returns:
    值加上常量后的稀疏张量（结构不变）
  """
  return tf.SparseTensor(
      indices=tensor.indices,
      values=constant + tensor.values,
      dense_shape=tensor.dense_shape)


def sparse_eye(size):
  """创建稀疏单位矩阵（对角线为 1，其余为 0）。
  
  使用稀疏表示（只存储对角线上的 size 个非零元素），内存效率高。
  
  Args:
    size: 矩阵大小（size × size 的方阵）
    
  Returns:
    size × size 的稀疏单位矩阵（SparseTensor）
  """
  indices = tf.cast(tf.stack([tf.range(size), tf.range(size)]), tf.int64)
  values = tf.ones(size)
  dense_shape = [tf.cast(size, tf.int64), tf.cast(size, tf.int64)]

  return tf.SparseTensor(
      indices=indices, values=values, dense_shape=dense_shape)


# modification from https://github.com/tensorflow/tensorflow/pull/21276
# without special initialization for g
class WeightNorm(tf.keras.layers.Wrapper):
  """Decouple weight magnitude and direction.

  This wrapper reparameterizes a layer by decoupling the weight's
  magnitude and direction. This speeds up convergence by improving the
  conditioning of the optimization problem.

  Weight Normalization: A Simple Reparameterization to Accelerate
  Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
  Tim Salimans, Diederik P. Kingma (2016)

  WeightNorm wrapper works for keras and tf layers.

  ```python
    net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
           input_shape=(32, 32, 3), data_init=True)(x)
    net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                     data_init=True)
    net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                     data_init=True)(net)
    net = WeightNorm(tf.keras.layers.Dense(n_classes),
                     data_init=True)(net)
  ```

  Arguments:
    layer: a layer instance.
    data_init: If `True` use data dependent variable initialization

  Raises:
    ValueError: If not initialized with a `Layer` instance.
    ValueError: If `Layer` does not contain a `kernel` of weights
    NotImplementedError: If `data_init` is True and running graph execution
  """

  def __init__(self, layer, data_init=False, **kwargs):
    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError(
          "Please initialize `WeightNorm` layer with a "
          "`Layer` instance. You passed: {input}".format(input=layer))

    super(WeightNorm, self).__init__(layer, **kwargs)
    self._track_trackable(layer, name="layer")

  def _compute_weights(self):
    """Generate weights with normalization."""
    with tf.variable_scope("compute_weights"):
      self.layer.kernel = tf.nn.l2_normalize(
          self.layer.v, axis=self.norm_axes) * self.layer.g

  def _init_norm(self, weights):
    """Set the norm of the weight vector."""
    with tf.variable_scope("init_norm"):
      flat = tf.reshape(weights, [-1, self.layer_depth])
      return tf.reshape(tf.norm(flat, axis=0), (self.layer_depth,))

  def _data_dep_init(self, inputs):
    """Data dependent initialization for eager execution."""

    with tf.variable_scope("data_dep_init"):
      # Generate data dependent init values
      activation = self.layer.activation
      self.layer.activation = None
      x_init = self.layer.call(inputs)
      m_init, v_init = tf.moments(x_init, self.norm_axes)
      scale_init = 1. / tf.sqrt(v_init + 1e-10)

    # Assign data dependent init values
    self.layer.g = self.layer.g * scale_init
    self.layer.bias = (-m_init * scale_init)
    self.layer.activation = activation
    self.initialized = True

  def build(self, input_shape=None):
    """Build `Layer`."""
    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = False

      if not hasattr(self.layer, "kernel"):
        raise ValueError("`WeightNorm` must wrap a layer that"
                         " contains a `kernel` for weights")

      # The kernel's filter or unit dimension is -1
      self.layer_depth = int(self.layer.kernel.shape[-1])
      self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

      self.layer.v = self.layer.kernel
      self.layer.g = self.layer.add_variable(
          name="g",
          shape=(self.layer_depth,),
          initializer=tf.ones_initializer,
          dtype=self.layer.kernel.dtype,
          trainable=True)

      # with ops.control_dependencies([self.layer.g.assign(
      #     self._init_norm(self.layer.v))]):
      #   self._compute_weights()
      self._compute_weights()

      self.layer.built = True
    self.input_spec = self.layer.input_spec

    super(WeightNorm, self).build()
    self.built = True

  def call(self, inputs):
    """Call `Layer`."""
    # if context.executing_eagerly():
    #   if not self.initialized:
    #     self._data_dep_init(inputs)
    self._compute_weights()  # Recompute weights for each forward pass

    output = self.layer.call(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())
