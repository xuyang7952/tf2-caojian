---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# 一、初识TensorFlow

## 1.1人工智能三学派

本讲目标:学会神经网络计算过程，使用基于TF2原生代码搭建你的第一个的神经网络训练模型

当今人工智能主流方向——连接 主义

- 前向传播
- 损失函数(初体会)
- 梯度下降(初体会)
- 学习率(初体会)
- 反向传播更新参数

人工智能：让机器具备人的思维和意识。

人工智能三学派:

- 行为主义:基于控制论，构建感知-动作控制系统。 (控制论，如平衡、行走、避障等自适应控制系统)
- 符号主义:基于算数逻辑表达式，求解问题时先把问题描述为表达式，再求解表达式。(可用公 式描述、实现理性思维，如专家系统)
- 连接主义:仿生学，模仿神经元连接关系。(仿脑神经元连接，实现感性思维，如神经网络)

理解:基于连结主义的神经网络设计过程

<img src="tensorflow2.assets/image-20200601174933788.png" alt="image-20200601174933788" style="zoom:67%;" />

用计算机仿出神经网络连接关系，让计算机具备感性思维。

- 准备数据:采集大量”特征标签”数据
- 搭建网络:搭建神经网络结构
- 优化参数:训练网络获取最佳参数(反传)
- 应用网络:将网络保存为模型，输入新数据，输出分类或预测结果(前传)

![image-20200508153816354](tensorflow2.assets/image-20200508153816354.png)

## 1.2神经网络设计过程

给鸢尾花分类(lris)

- 0狗尾草鸢尾
- 1杂色鸢尾
- 2弗吉尼亚鸢尾

人们通过经验总结出了规律:通过测量花的花曹长、花尊宽、花瓣长、花瓣宽，可以得出鸢尾花的类别。

(如:花萼长>花尊宽且花瓣长花瓣宽>2则为1杂色鸢尾)

if语句case语句——专家系统，把专家的经验告知计算机，计算机执行逻辑判别(理性计算)，给出分类。

**神经网络**：采集大量(输入特征：花萼长、花夢宽、花瓣长、花瓣宽，标签(需人工 标定)：对应的类别)数据对构成数据集。

把数据集限入搭建好的神经网络结构，网络优化参 数得到模型，模型读入新输入特征，输出识别‘

### 用神经网络实现鸢尾花分类

![image-20200601175024207](tensorflow2.assets/image-20200601175024207.png)

#### 喂入数据

![image-20200601175159359](tensorflow2.assets/image-20200601175159359.png)

#### 前向传播

![image-20200601175118615](tensorflow2.assets/image-20200601175118615.png)

代码示例：

```python
import tensorflow as tf

x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])  # 5.8,4.0,1.2,0.2（0）
w1 = tf.constant([[-0.8, -0.34, -1.4],
                  [0.6, 1.3, 0.25],
                  [0.5, 1.45, 0.9],
                  [0.65, 0.7, -1.2]])
b1 = tf.constant([2.52, -3.1, 5.62])
y = tf.matmul(x1, w1) + b1
print("x1.shape:", x1.shape)
print("w1.shape:", w1.shape)
print("b1.shape:", b1.shape)
print("y.shape:", y.shape)
print("y:", y)

#####以下代码可将输出结果y转化为概率值#####
y_dim = tf.squeeze(y)  # 去掉y中纬度1（观察y_dim与 y 效果对比）
y_pro = tf.nn.softmax(y_dim)  # 使y_dim符合概率分布，输出为概率值了
print("y_dim:", y_dim)
print("y_pro:", y_pro)

#请观察打印出的shape
```

#### 损失函数

![image-20200601175240914](tensorflow2.assets/image-20200601175240914.png)

得出结果为1分类

损失函数(loss function) ：预测值(y)与标准答案(y_ )的差距。

损失函数可以定量判断W、b的优劣，当损失函数输出最小时，参数w、b会出现最优值。

均方误差：
$$
\operatorname{MSE}\left(y, y_{-}\right)=\frac{\sum_{k=0}^{n}(y-y)^{2}}{n}
$$

#### 梯度下降

目的：想找到一组参数w和b,使得损失函数最小。

梯度：函数对各参数求偏导后的向量。函数梯度下降方向是函数减小方向。

梯度下降法：沿损失函数梯度下降的方向，寻找损失函数的最小值，得到最优参数的方法。
$$
\begin{array}{l}
w_{t+1}=w_{t}-l r * \frac{\partial l o s s}{\partial w_{t}} \\
b_{t+1}=b-l r * \frac{\partial l o s s}{\partial b_{t}} \\
w_{t+1} * x+b_{t+1} \rightarrow y
\end{array}
$$
学习率(learning rate, Ir) ：当学习率设置的过小时，收敛过程将变得十分缓慢。而当学习率设置的过大时，梯度可能会在最小值附近来回震荡，甚至可能无法收敛。

![image-20200601175413420](tensorflow2.assets/image-20200601175413420.png)

#### 反向传播

反向传播：从后向前，逐层求损失函数对每层神经元参数的偏导数，迭代更新所有参数。
$$
w_{t+1}=w_{t}-l r * \frac{\partial l o s s}{\partial w_{t}}
$$
例如损失函数为：
$$
\operatorname{loss}=(w+1)^{2}
$$
求偏导
$$
\frac{\partial \operatorname{loss}}{\partial w}=2 w+2
$$
参数w初始化为5，学习率为0.2则

| 次数 | 参数w | 结果                          |
| ---- | ----- | ----------------------------- |
| 1    | 5     | `5-0.2*（2*4+2）=2.6`         |
| 2    | 2.6   | `2.6-0.2*（2*2.6+2）=1.16`    |
| 3    | 1.16  | `1.16-0.2*（2*1.16+2）=0.296` |
| 4    | 0.296 | .......                       |

![image-20200601175508405](tensorflow2.assets/image-20200601175508405.png)

求出w的最佳值，使得损失函数最小，代码示例：

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))
# print(w)
lr = 0.2
epoch = 20

for i in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环40次迭代。
    # 用with结构让损失函数loss对参数w求梯度
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导,此处为loss函数对w求偏导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f" % (i, w.numpy(), loss))

# lr初始值：0.2   请自改学习率  0.001  0.999 看收敛过程
# 最终目的：找到 loss 最小 即 w = -1 的最优参数w
```

## 1.3张量生成

张量(Tensor) ：多维数组(列表)

阶：张量的维数

| 维数 | 阶   | 名字        | 例子                              |
| ---- | ---- | ----------- | --------------------------------- |
| 0-D  | 0    | 标量 scalar | s=1 2 3                           |
| 1-D  | 1    | 向量 vector | v=[1, 2, 3]                       |
| 2-D  | 2    | 矩阵 matrix | m=`[[1,2,3],[4,5,6][7,8,9]]`      |
| N-D  | N    | 张量 tensor | t=[[[    有几个中括号就是几阶张量 |

张量可以表示0阶到n阶数组(列表)

### 数据类型

- `tf.int`, `tf.float`
  - `tf.int 32`，`tf.float 32`, `tf.float 64`
- `tf.bool`
  - `tf.constant([True, False])`
- `tf.string`
  - `tf.constant("Hello, world!")`

### 如何创建一个Tensor

**创建一个张量**

```
tf.constant(张量内容, dtype=数据类型（可选）)
```

代码示例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭log信息

import tensorflow as tf

a = tf.constant([1, 5], dtype=tf.int64)
b = tf.constant(1)
c = tf.constant([[1,2],[2,3]])
d = tf.constant([[[1,2],[2,3]],[[1,2],[2,3]]])
print("b",b)
print("a:", a)
print("c:", c)
print("d:", d) # 张量的维度和shape中的“,”分隔开的数字有关，未分开，就是0维，分开一个就是1维，分开两个就是2维，
print("a.dtype:", a.dtype)
print("a.shape:", a.shape)

# 本机默认 tf.int32  可去掉dtype试一下 查看默认值
```

将numpy的数据类型转换为Tensor数据类型tf. convert to_tensor(数据名，dtype=数 据类型(可选))

代码示例

```python
import tensorflow as tf
import numpy as np

a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print("a:", a)
print("b:", b)
```

- **创建全为0的张量**
  - `tf. zeros(维度)`
- **创建全为1的张量**
  - `tf. ones(维度)`
- **创建全为指定值的张量**
  - `tf. fil(维度，指定值)`

维度：

- 一维直接写个数
- 二维用[行，列]
- 多维用[n,m,.K....]

代码示例：

```python
import tensorflow as tf

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print("a:", a)
print("b:", b)
print("c:", c)
```

- **生成正态分布的随机数，默认均值为0，标准差为1**
  - `tf. random.normal (维度，mean=均值，stddev=标准差)`
- **生成截断式正态分布的随机数**
  - `tf. random.truncated normal (维度，mean=均值，stddev=标准差)`

在`tf.truncated normal`中如果随机生成数据的取值在(μ-2σ， μ+2σ) 之外,则重新进行生成，保证了生成值在均值附近。(μ:均值，σ:标准差）

标准差计算公式：
$$
\sigma=\sqrt{\frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}}{n}}
$$
代码示例：

```python
import tensorflow as tf

d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)
```

生成均匀分布随机数( minval, maxval ) tf. random. uniform(维度，minval=最小值，maxval=最大值)

代码示例

```python
import tensorflow as tf

f = tf.random.uniform([2, 2], minval=0, maxval=1,seed=11)
print("f:", f)
```

## 1.4 TF2常用函数

- 强制tensor转换为该数据类型
  - `tf.cast (张量名，dtype=数据类型)`
- 计算张量维度上元素的最小值
  - `tf.reduce_ min (张量名)`
- 计算张量维度上元素的最大值
  - `tf.reduce_ max(张量名)`
- 将tensor转换为numpy
  - `tensor.numpy()`

代码示例：

```python
import tensorflow as tf

x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1:", x1)
x2 = tf.cast(x1, tf.int32)
print("x2:", x2)
print("minimum of x2：", tf.reduce_min(x2))
print("maxmum of x2:", tf.reduce_max(x2))

a = tf.constant(5, dtype=tf.int64)
print("tensor a:", a)
print("numpy a:", a.numpy())
```

### 理解axis

在一个二维张量或数组中，可以通过调整axis等于0或1控制执行维度。

axis=0代表跨行(经度，down),而axis=1代表跨列(纬度，across)

如果不指定axis,则所有元素参与计算。

![image-20200601175653922](tensorflow2.assets/image-20200601175653922.png)

- 计算张量沿着指定维度的平均值
  - `tf.reduce_mean (张量名，axis=操作轴)`
- 计算张量沿着指定维度的和
  - `tf.reduce_sum (张量名，axis=操作轴)`

代码示例

```python
x = tf.constant([[1, 2, 3], [2, 2, 3]])
print("x:", x)
print("mean of x:", tf.reduce_mean(x))  # 求x中所有数的均值
print("sum of x:", tf.reduce_sum(x, axis=1))  # 求每一行的和
```

### 变量`tf.Variable`

`tf.Variable ()` 将变量标记为“可训练”，被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练参数。

`tf.Variable(初始值)`

定义变量

```python
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
print(w)
tf.print(w)
```

### TensorFlow中的数学运算

- 对应元素的四则运算: `tf.add`, `tf.subtract`, `tf.multiply`, `tf.divide`
- 平方、次方与开方: `tf.square`, `tf.pow`, `tf.sqrt`
- 矩阵乘: `tf.matmul`

**对应元素的四则运算**

- 实现两个张量的对应元素相加
  - `tf.add (张量1，张量2)`
- 实现两个张量的对应元素相减
  - `tf.subtract (张量1，张量2)`
- 实现两个张量的对应元素相乘
  - `tf.multiply (张量1,张量2)`
- 实现两个张量的对应元素相除
  - `tf.divide (张量1，张量2)`

注意：只有维度相同的张量才可以做四则运算

代码示例：

```python
import tensorflow as tf

a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
print("a:", a)
print("b:", b)
print("a+b:", tf.add(a, b))
print("a-b:", tf.subtract(a, b))
print("a*b:", tf.multiply(a, b))
print("b/a:", tf.divide(b, a))

```

**平方、次方与开方**

- 计算某个张量的平方
  - `tf.square (张量名)`
- 计算某个张量的n次方
  - `tf.pow (张量名，n次方数)`
- 计算某个张量的开方
  - `tf.sqrt (张量名)`

代码示例：

```python
import tensorflow as tf

a = tf.fill([1, 2], 3.)
print("a:", a)
print("a的次方:", tf.pow(a, 3))
print("a的平方:", tf.square(a))
print("a的开方:", tf.sqrt(a))
```

### 矩阵乘

实现两个矩阵的相乘

`tf.matmul(矩阵1，矩阵2)`

代码示例：

```python
import tensorflow as tf

a = tf.ones([3, 2],dtype=tf.int32)
b = tf.fill([2, 3], 3,)
print("a:", a)
print("b:", b)
print("a*b:", tf.matmul(a, b))
```

### 传入特征与标签

切分传入张量的第一维度， 生成输入特征标签对，构建数据集

`data = tf.data.Dataset.from_tensor_ slices((输入特征，标签))`

(Numpy和Tensor格式都可用该语句读入数据)

代码示例：

```python
import tensorflow as tf

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print("dataset:",dataset,"\n")
for element in dataset:
    print(element)
```

### 函数对指定参数求导`gradient`

with结构记录计算过程，gradient求 出张量的梯度


代码示例，对函数x^2求x的导数
$$
\frac{\partial w^{2}}{\partial w}=2 w=2^{*} 3.0=6.0
$$

```python
import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))
    y = tf.pow(x, 2)
grad = tape.gradient(y, x)
print(grad)
```

### 枚举元素`enumerate`

enumerate是python的内建函数，它可遍历每个元素(如列表、元组或字符串)，

组合为：索引元素，常在for循环中使用。

`enumerate(列表名)`

代码示例：

```python
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)
```

### 独热编码

独热编码(Cone-hot encoding) ：在分类问题中，常用独热码做标签，标记类别: 1表示是，0表示非。

标签为1，独热编码为

| 0狗尾草鸢尾 | 1杂色鸢尾 | 2弗吉尼亚鸢尾 |
| ----------- | --------- | ------------- |
| 0           | 1         | 0             |

`tf.one_ hot()`函数将待转换数据，转换为one-hot形式的数据输出。

`tf.one_ hot (待转换数据，depth=几分类)`

代码示例：

```python
import tensorflow as tf

classes = 3
labels = tf.constant([1, 0, 2])  # 输入的元素值最小为0，最大为2
output = tf.one_hot(labels, depth=classes)
print("labels",labels)
print("result of labels1:", output)
print("\n")
```

### 输出符合概率分布（归一化）

`tf.nn.softmax(x)`把一个N*1的向量归一化为（0，1）之间的值，由于其中采用指数运算，使得向量中数值较大的量特征更加明显。
$$
\operatorname{Softmax}\left(y_{i}\right)=\frac{e^{y_{i}}}{\sum_{j=0}^{n} e^{y_{i}}}
$$
`tf.nn.softmax(x)`使输出符合概率分布

当n分类的n个输出(yo, y1...... yn_1)通过`softmax( )`函数，便符合概率分布，所有的值和为1。
$$
\forall x P(X=x) \in[0,1] \text { 且 } \sum_{x} P(X=x)=1
$$
![image-20200601180611527](tensorflow2.assets/image-20200601180611527.png)

代码示例：

```python
import tensorflow as tf

y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)

print("After softmax, y_pro is:", y_pro)  # y_pro 符合概率分布,

print("The sum of y_pro:", tf.reduce_sum(y_pro))  # 通过softmax后，所有概率加起来和为1
```

### 参数自更新

赋值操作，更新参数的值并返回。

调用`assign_ sub`前，先用`tf.Variable`定义变量w为可训练(可自更新)。

`w.assign_ sub (w要自减的内容)`

代码示例：

```python
import tensorflow as tf

x = tf.Variable(4)
x.assign_sub(1)
print("x:", x)  # 4-1=3
```

### 指定维度最大值索引

返回张量沿指定维度最大值的索引

`tf.argmax (张量名,axis=操作轴)`

代码示例：

```python
import numpy as np
import tensorflow as tf

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print("test:\n", test)
print("每一列的最大值的索引：", tf.argmax(test, axis=0))  # 返回每一列最大值的索引
print("每一行的最大值的索引", tf.argmax(test, axis=1))  # 返回每一行最大值的索引
```

## 1.5鸢尾花数据集读入

**数据集介绍**

共有数据150组，每组包括花尊长、花尊宽、花瓣长、花瓣宽4个输入特征。同时给出了，这组特征对应的鸢尾花 类别。类别包括Setosa Iris (狗尾草鸢尾)，Versicolour lris (杂色鸢尾)，Virginica Iris (弗吉尼亚鸢尾)三类，分别用数字0，1，2表示。

读取数据集代码

```python
from sklearn import datasets
from pandas import DataFrame
import pandas as pd

x_data = datasets.load_iris().data  # .data返回iris数据集所有输入特征
y_data = datasets.load_iris().target  # .target返回iris数据集所有标签
print("x_data from datasets: \n", x_data[:5])
print("y_data from datasets: \n", y_data[:5])


x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']) # 为表格增加行索引（左侧）和列标签（上方）
pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
print("x_data add index: \n", x_data.head())

x_data['类别'] = y_data  # 新加一列，列标签为‘类别’，数据为y_data
print("x_data add a column: \n", x_data.head())

#类型维度不确定时，建议用print函数打印出来确认效果
```

## 1.6神经网络实现鸢尾花分类

1. 准备数据
   - 数据集读入
   - 数据集乱序
   - 生成训练集和测试集(即x_ _train/y_ train, x_ test/y_ test)
   - 配成(输入特征，标签)对，每次读入一小撮(batch)
2. 搭建网络
   - 定义神经网路中所有可训练参数
3. 参数优化
   - 嵌套循环迭代，with结构更新参数，显示当前loss
4. 测试效果
   - 计算当前参数前向传播后的准确率，显示当前acc
5. acc / loss可视化

代码示例：

```python
# 导入所需模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

print(x_train[:5])
print(y_train[:5])
```

```python
print(len(x_data),len(x_train),len(x_test))
```

```python
# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）30*4，
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(30)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(30)
print(train_db)
for item in train_db.take(1):
    print(item)
```

```python
# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 200  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和
```

```python
# 训练部分
for epoch in range(epoch):  #数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  
        #batch级别的循环 ，每个step对应一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）  
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            
            loss = tf.reduce_mean(tf.square(y_ - y))   # 采用均方误差损失函数mse = mean(sum(y_ - y)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
#         print("step",step)
#         print("x_train",x_train)
#         print("y_train",y_train)
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新



    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    
    if epoch%25 ==0:
        # 打印loss信息
        print("Epoch {}, loss: {}".format(epoch, loss_all/4))
        print("Test_acc:", acc)
        print("--------------------------")
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备
```

```python
print(train_loss_results[:5])
print(test_acc[:5])
```

```python
# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Ac sc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
```

#### 神经网络实现鸢尾花分类--mycode

```python
from sklearn.model_selection import train_test_split
from sklearn import datasets,metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python
# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
# x_data = pd.DataFrame(datasets.load_iris().data)  pandas 就不能用了
# y_data = pd.DataFrame(datasets.load_iris().target)
print(x_data[:5])
print(y_data[:5])
```

```python
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
print(X_train.shape,X_test.shape)
```

```python
X_train = tf.cast(X_train,tf.float32)
X_test = tf.cast(X_test,tf.float32)
```

```python
ds_train = tf.data.Dataset.from_tensor_slices((X_train,y_train)).batch(30)
ds_test = tf.data.Dataset.from_tensor_slices((X_test,y_test)).batch(30)
print(ds_train)
```

```python
for i ,(x,y) in enumerate(ds_train):
    print(i)
    print(x)
    print(y)
    print("====="*20)
```

```python
# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
print(w1)
print(b1)
```

```python
lr = 0.2  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 100  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和
```

```python
# 训练部分
for epoch in range(epoch):  #数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  
        #batch级别的循环 ，每个step对应一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）  
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            
            loss = tf.reduce_mean(tf.square(y_ - y))   # 采用均方误差损失函数mse = mean(sum(y_ - y)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
#         print("step",step)
#         print("x_train",x_train)
#         print("y_train",y_train)
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新



    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    
    if epoch%25 ==0:
        # 打印loss信息
        print("Epoch {}, loss: {}".format(epoch, loss_all/4))
        print("Test_acc:", acc)
        print("--------------------------")
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备
```

```python
# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Ac sc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
```

# 二、深入TensorFlow

## 预备知识

### `tf.where()`

条件语句真返回A,条件语句假返回B

`tf.where(条件语句，真返回A，假返回B)`

代码示例

```python
import tensorflow as tf

a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
print("c：", c)
```

### `np.random.RandomState.rand()`

返回一个[0,1)之间的随机数
np.random.RandomState.rand(维度)  # 若维度为空，返回标量

```python
rdm = np.random.RandomState(seed=1)
a = rdm.rand()
b = rdm.rand(2, 3)
print("a:", a)
print("b:", b)
```

### `np.vstack()`

将两个数组按垂直方向叠加

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print("c:\n", c)
```

### 生成网格坐标点

- `np.mgrid[ ]`
  - `np.mgrid[起始值:结束值:步长，起始值:结束值:步长，... ]`
  - [起始值，结束值)，区间左闭右开
- `x.ravel()`将x变为一维数组，“把.前变量拉直”
- `np.c_[]` 使返回的间隔数值点配对
  - `np.c_ [数组1，数组2，... ]`

代码示例：

```python
# 生成等间隔数值点
x, y = np.mgrid[1:3:1, 2:4:0.5]
# 将x, y拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[x.ravel(), y.ravel()]
print("x:\n", x)
print("y:\n", y)
print("x.ravel():\n", x.ravel())
print("y.ravel():\n", y.ravel())
print('grid:\n', grid)
```

```python
a = np.mgrid[1:3:1]
b = np.mgrid[1:3:1,2:4:0.5]
c = np.mgrid[1:3:1,2:4:1,2:4:0.5]
print(a.shape,b.shape,c.shape)
print(a)
print(b)
print(c)
```

`np.mgrid[起始值:结束值:步长，起始值:结束值:步长]`填入两个值，相当于构建了一个二维坐标，很坐标值为第一个参数，纵坐标值为第二个参数。

例如，横坐标值为[1, 2, 3]，纵坐标为[2, 2.5, 3, 3.5]，
这样x和y都为3行4列的二维数组，每个点一一对应构成一个二维坐标区域

```python
x, y = np.mgrid[1:5:1, 2:4:0.5]
print("x:\n", x)
print("y:\n", y)
```

## 复杂度和学习率

### 神经网络复杂度

<img src="tensorflow2.assets/image-20200601183556407.png" alt="image-20200601183556407" style="zoom:67%;" />

NN复杂度：多用NN层数和NN参数的个数表示

**空间复杂度:**

层数=隐藏层的层数+ 1个输出层

图中为：2层NN

总参数=总w+总b

第1层：3x4+4

第2层：4x2+2

图中共计：3x4+4 +4x2+2 = 26

**时间复杂度:**

乘加运算次数

第1层：3x4

第2层：4x2

图中共计：3x4 + 4x2 = 20

### 学习率

$$
w_{t+1}=w_{t}-l r * \frac{\partial l o s s}{\partial w_{t}}
$$

参数说明

- 更新后的参数
- 当前参数
- 学习率
- 损失函数的梯度（偏导数）

**指数衰减学习率**

可以先用较大的学习率，快速得到较优解，然后逐步减小学习率，使模型在训练后期稳定。
指数衰减学习率=初始学习率*学习率衰减率(当前轮数1多少轮衰减一次)

![image-20200511162438441](tensorflow2.assets/image-20200511162438441.png)

代码示例

```python
import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))

epoch = 20
LR_BASE = 0.2  # 最初学习率
LR_DECAY = 0.99  # 学习率衰减率
LR_STEP = 2  # 喂入多少轮BATCH_SIZE后，更新一次学习率

for epoch in range(epoch):  
# for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环40次迭代。
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)  
    # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    if epoch%2 ==0:
        print("After %s epoch,w is %f,loss is %f,lr is %f" % (epoch, w.numpy(), loss, lr))
```

## 激活函数

为什么要用激活函数：在神经网络中，如果不对上一层结点的输出做非线性转换的话，再深的网络也是线性模型，只能把输入线性组合再输出，不能学习到复杂的映射关系，因此需要使用激活函数这个非线性函数做转换。

参考：https://www.cnblogs.com/itmorn/p/11132494.html

### Sigmoid函数

$$
\begin{aligned}
&\operatorname{sigmod}(x)=\frac{1}{1+e^{-x}} \in(0,1)\\
&\operatorname{sigmod}^{\prime}(x)=\operatorname{sigmod}(x)^{*}(1-\operatorname{sigmod}(x))=\frac{1}{1+e^{-x}} * \frac{e^{-x}}{1+e^{-x}}=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}} \in(0,0.25)
\end{aligned}
$$

```python
```tf.nn.sigmoid(x)```
```

sigmoid函数图像

![img](tensorflow2.assets/v2-15ef91c7563ef2a046de444a904f1ff8_720w.jpg)

sigmoid导数图像

![img](tensorflow2.assets/v2-4b322e9c5d48a434c8a400d96a1de5fd_720w.jpg)

目前使用sigmoid函数为激活函数的神经网络已经很少了

特点

(1)易造成梯度消失

​		深层神经网络更新参数时，需要从输入层到输出层，逐层进行链式求导，而sigmoid函数的导数输出为[0,0.25]间的小数，链式求导需要多层导数连续相乘，这样会出现多个[0,0.25]间的小数连续相乘，从而造成结果趋于0，产生梯度消失，使得参数无法继续更新。

(2)输出非0均值，收敛慢

​		希望输入每层神经网络的特征是以0为均值的小数值，但sigmoid函数激活后的数据都时整数，使得收敛变慢。

(3)幂运算复杂，训练时间长

​		sigmoid函数存在幂运算，计算复杂度大。

### Tanh函数

$$
\begin{array}{l}
\tanh (x)=\frac{1-e^{-2 x}}{1+e^{-2 x}} \in(-1,1) \\
\tanh ^{\prime}(x)=1-(\tanh (x))^{2}=\frac{4 e^{-2 x}}{\left(1+e^{-2 x}\right)^{2}} \in(0,1]
\end{array}
$$


```tf.math.tanh(x)```


![image-20200601183826652](tensorflow2.assets/image-20200601183826652.png)

**特点**

(1)输出是0均值

(2)易造成梯度消失

(3)幂运算复杂，训练时间长

### Relu函数

$$
\begin{array}{l}
r e l u(x)=\max (x, 0)=\left\{\begin{array}{l}
x, \quad x \geq 0 \\
0, \quad x<0
\end{array} \in[0,+\infty)\right. \\
r e l u^{\prime}(x)=\left\{\begin{array}{ll}
1, & x \geq 0 \\
0, & x<0
\end{array} \in\{0,1\}\right.
\end{array}
$$

```
tf.nn.relu(x)
```



![image-20200601183848839](tensorflow2.assets/image-20200601183848839.png)

**优点:**

1. 解决了梯度消失问题(在正区间)
2. 只 需判断输入是否大于0，计算速度快
3. 收敛速度远快于sigmoid和tanh

**缺点:**

1. 输出非0均值，收敛慢
2. Dead ReIU问题:某些神经元可能永远不会被激活，导致相应的参数永远不能被更新。

### Leaky Relu函数

$$
\begin{aligned}
&\text { LeakyReLU }(x)=\left\{\begin{array}{ll}
x, & x \geq 0 \\
a x, & x<0
\end{array} \in R\right.\\
&\text { LeakyReL } U^{\prime}(x)=\left\{\begin{array}{ll}
1, & x \geq 0 \\
a, & x<0
\end{array} \in\{a, 1\}\right.
\end{aligned}
$$


``` tf.nn.leaky_relu(x)``` 
![image-20200601183910439](tensorflow2.assets/image-20200601183910439.png)

理论上来讲，Leaky Relu有Relu的所有优点，外加不会有Dead Relu问题，但是在实际操作当中，并没有完全证明Leaky Relu总是好于Relu。

### 总结

- 首选relu激活函数;
- 学习率设置较小值;
- 输入特征标准化，即让输入特征满足以0为均值，1为标准差的正态分布;
- 初始参数中心化，即让随机生成的参数满足以0为均值，下式为标准差的正态分布

$$
\sqrt{\frac{2}{\text { 当前层输入特征个数 }}}
$$

## 损失函数

损失函数(loss) ：预测值(y) 与已知答案(y_) 的差距

NN优化目标：loss最小，有三种方法

- mse (Mean Squared Error)
- 自定义
- ce (Cross Entropy)

### 均方误差mes

$$
\operatorname{MSE}\left(y_{-}, y\right)=\frac{\sum_{i=1}^{n}\left(y-y_{-}\right)^{2}}{n}
$$

```
loss_mse = tf.reduce_mean(tf.square(y_ - y))
```

预测酸奶日销量y, x1、 x2是影响日销量的因素。

建模前，应预先采集的数据有:每日x1、x2和销量y_ (即已知答案，最佳情况:产量=销量)

拟造数据集X，Y_ : y_ =x1 + x2，噪声: -0.05~ +0.05

拟合可以预测销量的函数

代码示例

```python
import tensorflow as tf
import numpy as np

SEED = 23455

rdm = np.random.RandomState(seed=SEED) # 生成[0,1)之间的随机数
x = rdm.rand(32,2)
y_1 =[x1 + x2 + (rdm.rand()/10.0-0.05) for (x1,x2) in x] # 生成噪声[0,1)/10 -0.05=[-0.05,0.05);
y_ =[[x1 + x2 + (rdm.rand()/10.0-0.05)] for (x1,x2) in x] # 生成噪声[0,1)/10 -0.05=[-0.05,0.05);
print(x[:5])
print(y_1[:5])
print(y_[:5])
```

```python
y_1 =[x1 + x2   for (x1,x2) in x] # 生成噪声[0,1)/10 -0.05=[-0.05,0.05);
y_ =[[x1 + x2 ] for (x1,x2) in x] # 生成噪声[0,1)/10 -0.05=[-0.05,0.05);
print(y_1[:5])
print(y_[:5])
```

```python
x = tf.cast(x,dtype=tf.float32)
w1 = tf.Variable(tf.random.normal([2,1],seed=1))
print(w1)
```

```python
epochs = 200
lr = 0.4

for epoch in range(epochs):
    with tf.GradientTape()as tape:
        y = tf.matmul(x,w1)
        loss_mse = tf.reduce_mean(tf.square(y-y_))
    grads = tape.gradient(loss_mse,w1)
    w1.assign_sub(grads*lr)
    if epoch % 10 == 0:
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
```

### 自定义损失函数

如预测商品销量，预测多了，损失成本;预测少了，损失利润

若利润≠成本，则mse产生的loss无法利益最大化。

自定义损失函数，y_：标准答案数据集的，y：预测答案计算出的
$$
\operatorname{loss}\left(y_{-} y\right)=\sum_{n} f\left(y_， y\right)
$$

$$
f\left(y_{-}, y\right)=\left\{\begin{array}{lll}\text { PROFIT* }\left(y_{-}-y\right) & y<y_{-} & \text {预测的 } y\text { 少了, 损失利高(PROFIT) } \\ \text { COST } *\left(y-y_{-}\right) & y>=y_{-} & \text {预测的 } y \text { 多了，损失成本(COST) }\end{array}\right.
$$

如:预测酸奶销量，酸奶成本(COST) 1元，酸奶利润(PROFIT) 99元。

预测少了损失利润99元，大于预测多了损失成本1元。预测少了损失大，希望生成的预测函数往多了预测。

则损失函数为：loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))

```python
import tensorflow as tf
import numpy as np

SEED = 23455
COST = 1
PROFIT = 99

rdm = np.random.RandomState(SEED)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 1000
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))

    grads = tape.gradient(loss, w1)
    w1.assign_sub(lr * grads)

    if epoch % 100 == 0:
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
```

自定义损失函数，酸奶成本99元， 酸奶利润1元，成本很高，利润很低，人们希望多少预测，生成模型系数小于1，往少了预测。运行结果

```python
import tensorflow as tf
import numpy as np

SEED = 23455
COST = 99
PROFIT = 1

rdm = np.random.RandomState(SEED)
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x]  # 生成噪声[0,1)/10=[0,0.1); [0,0.1)-0.05=[-0.05,0.05)
x = tf.cast(x, dtype=tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 1000
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * COST, (y_ - y) * PROFIT))

    grads = tape.gradient(loss, w1)
    w1.assign_sub(lr * grads)

    if epoch % 100 == 0:
        print("After %d training steps,w1 is " % (epoch))
        print(w1.numpy(), "\n")
print("Final w1 is: ", w1.numpy())
```

### 交叉熵损失函数

交义熵损失函数CE (Cross Entropy)：表征两个概率分布之间的距离
$$
\mathrm{H}\left(\mathrm{y}_{-}, \mathrm{y}\right)=-\sum y_{-} * \ln y
$$
eg.二分类已知答案y_ =(1, 0)，预测y1=(0.6, 0.4) y2=(0.8, 0.2)哪个更接近标准答案?
$$
\begin{aligned}
&\mathrm{H}_{1}((1,0),(0.6,0.4))=-(1 * \ln 0.6+0 * \ln 0.4) \approx-(-0.511+0)=0.511\\
&\mathrm{H}_{2}((1,0),(0.8,0.2))=-(1 * \ln 0.8+0 * \ln 0.2) \approx-(-0.223+0)=0.223
\end{aligned}
$$
因为H> H2，所以y2预测更准：tf.losses.categorical crossentropy(y_ ，y)

```python
import tensorflow as tf

loss_ce1 = tf.losses.categorical_crossentropy([1, 0], [0.6, 0.4])
loss_ce2 = tf.losses.categorical_crossentropy([1, 0], [0.8, 0.2])
print("loss_ce1:", loss_ce1)
print("loss_ce2:", loss_ce2)

# 交叉熵损失函数
```

### 交叉熵损失函数与softmax结合

输出先过softmax函数，再计算y与y_ 的交叉熵损失函数。
tf.nn.softmax_cross_entropy_with_logits(y_, y)

```python
# softmax与交叉熵损失函数的结合
import tensorflow as tf
import numpy as np

y_ = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
y = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
y_pro = tf.nn.softmax(y)
print("y_pro:",y_pro)
loss_ce1 = tf.losses.categorical_crossentropy(y_,y_pro)
loss_ce2 = tf.nn.softmax_cross_entropy_with_logits(y_, y)

tf.print('分步计算的结果:\n', loss_ce1)
tf.print('结合计算的结果:\n', loss_ce2)
```

## 过拟合与欠拟合

![image-20200601183952319](tensorflow2.assets/image-20200601183952319.png)

欠拟合的解决方法:

- 增加输入特征项
- 增加网络参数
- 减少正则化参数

过拟合的解决方法:

- 数据清洗
- 增大训练集
- 采用正则化
- 增大正则化参数

### 正则化缓解过拟合

正则化在损失函数中引入模型复杂度指标，利用给W加权值，弱化了训练数据的噪声(一般不正则化b)
$$
\operatorname{loss}=\operatorname{loss}\left(\mathrm{y}与{y}_{-}\right)+\mathrm{REGULARIZER}{*} \operatorname{loss}(\mathrm{w})
$$
式中含义：

`loss(y与y_)`：模型中所有参数的损失函数。如:交叉熵、均方误差

`REGULARIZER`：用超参数REGULARIZER给出参数w在总loss中的比例，即正则化的权重

`loss(w)`：需要正则化的参数。计算方式有两种
$$
\operatorname{loss}_{L_{1}}(w)=\sum_{i}\left|w_{i}\right|
$$

$$
\operatorname{loss}_{L 2}(w)=\sum_{i}\left|w_{i}^{2}\right|
$$

正则化的选择

- L1正则化大概率会使很多参数变为零，因此该方法可通过稀疏参数，即减少参数的数量，降低复杂度。
- L2正则化会使参数很接近零但不为零，因此该方法可通过减小参数值的大小降低复杂度。
  - `tf.nn.l2_loss(w)`

代码示例，未采用正则化`p29_regularizationfree.py`

```python
# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据/标签 生成x_train y_train
df = pd.read_csv('./data/dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

# reshape(-1,x) -1是将一维数组转换为二维的矩阵，并且第二个参数是表示分成几列，
# 但是在reshape的时候必须让数组里面的个数和shape的函数做取余时值为零才能转换
x_train = np.vstack(x_data).reshape(-1,2)
y_train = np.vstack(y_data).reshape(-1,1)  #将y_data转换为二维数组


Y_c = [['red' if y else 'blue'] for y in y_train]  # 三元运算

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数，输入层为2个神经元，隐藏层为11个神经元，1层隐藏层，输出层为1个神经元
# 隐藏层11个神经元为人为指定
# 用tf.Variable()保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)  # 隐藏层2个输入，11个输出
b1 = tf.Variable(tf.constant(0.01, shape=[11]))  # b的个数与w个数相同

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)  # 输出层接收11个，输出1个
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01  # 学习率
epoch = 200  # 循环轮数

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)  # relu激活函数
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss = tf.reduce_mean(tf.square(y_train - y))

        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        # w1 = w1 - lr * w1_grad tape.gradient是自动求导结果与[w1, b1, w2, b2] 索引为0，1，2，3 
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每20个epoch，打印loss信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx , yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_test in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c)) # squeeze去掉纬度是1的纬度,相当于去掉[['red'],['blue']],内层括号变为['red','blue']
# 把坐标xx yy和对应的值probs放入contour<[‘kɑntʊr]>函数，给probs值为0.5的所有点上色  plt点show后 显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])  # 画出probs值为0.5轮廓线,levels:这个参数用于显示具体哪几条登高线
plt.show()

# 读入红蓝点，画出分割线，不包含正则化
# 不清楚的数据，建议print出来查看
```

```python
# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据/标签 生成x_train y_train
df = pd.read_csv('./data/dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = x_data
y_train = y_data.reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数，输入层为4个神经元，隐藏层为32个神经元，2层隐藏层，输出层为3个神经元
# 用tf.Variable()保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01  # 学习率为
epoch = 400  # 循环轮数

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_mse = tf.reduce_mean(tf.square(y_train - y))
            # 添加l2正则化
            loss_regularization = []
            # tf.nn.l2_loss(w)=sum(w ** 2) / 2
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))
            # 求和
            # 例：x=tf.constant(([1,1,1],[1,1,1]))
            #   tf.reduce_sum(x)
            # >>>6
            # loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + 0.03 * loss_regularization # REGULARIZER = 0.03

        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每200个epoch，打印loss信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_predict in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_predict], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
# 把坐标xx yy和对应的值probs放入contour<[‘kɑntʊr]>函数，给probs值为0.5的所有点上色  plt点show后 显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 读入红蓝点，画出分割线，包含正则化
# 不清楚的数据，建议print出来查看 
```

```python
# 导入所需模块
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# 读入数据/标签 生成x_train y_train
df = pd.read_csv('./data/dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = x_data
y_train = y_data.reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型问题报错
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# from_tensor_slices函数切分传入的张量的第一个维度，生成相应的数据集，使输入特征和标签值一一对应
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数，输入层为4个神经元，隐藏层为32个神经元，2层隐藏层，输出层为3个神经元
# 用tf.Variable()保证参数可训练
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01  # 学习率为
epoch = 400  # 循环轮数

# 训练部分
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息

            h1 = tf.matmul(x_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_mse = tf.reduce_mean(tf.square(y_train - y))
            # 添加l2正则化
            loss_regularization = []
            # tf.nn.l2_loss(w)=sum(w ** 2) / 2
            loss_regularization.append(tf.nn.l2_loss(w1))
            loss_regularization.append(tf.nn.l2_loss(w2))
            # 求和
            # 例：x=tf.constant(([1,1,1],[1,1,1]))
            #   tf.reduce_sum(x)
            # >>>6
            # loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            loss_regularization = tf.reduce_sum(loss_regularization)
            loss = loss_mse + 0.06 * loss_regularization # REGULARIZER = 0.03

        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    # 每200个epoch，打印loss信息
    if epoch % 20 == 0:
        print('epoch:', epoch, 'loss:', float(loss))

# 预测部分
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_predict in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_predict], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
# 把坐标xx yy和对应的值probs放入contour<[‘kɑntʊr]>函数，给probs值为0.5的所有点上色  plt点show后 显示的是红蓝点的分界线
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

# 读入红蓝点，画出分割线，包含正则化
# 不清楚的数据，建议print出来查看 
```

```python

```
