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

# 三、使用八股搭建神经网络

## 搭建网络sequenial

用Tensorflow API: `tf. keras`

六步法搭建神经网络

- 第一步：import相关模块，如import tensorflow as tf。
- 第二步：指定输入网络的训练集和测试集，如指定训练集的输入x_train和标签y_train，测试集的输入x_test和标签y_test。
- 第三步：逐层搭建网络结构，model = tf.keras.models.Sequential()。
- 第四步：在model.compile()中配置训练方法，选择训练时使用的优化器、损失函数和最终评价指标。
- 第五步：在model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个batch的大小（batchsize）和数据集的迭代次数（epoch）。
- 第六步：使用model.summary()打印网络结构，统计参数数目。

### Sequential()容器

Sequential()可以认为是个容器，这个容器里封装了一个神经网络结构。

model = tf.keras.models.Sequential ([网络结构]) #描述各层网络

在Sequential()中，要描述从输入层到输出层每一层的网络结构。每一层的网络结构可以是：

- 拉直层：`tf.keras.layers.Flatten( )`
  - 这一层不含计算，只是形状转换，把输入特征拉直变成一维数组
- 全连接层：`tf.keras.layers.Dense(神经元个数，activation= "激活函数“，kernel_regularizer=哪种正则化)`
  - activation (字符串给出)可选: relu、softmax、sigmoid、tanh
  - kernel_regularizer可 选: `tf.keras.regularizers.l1()`、 `tf.keras.regularizers.12()`
- 卷积层：`tf.keras.layers.Conv2D(filters =卷积核个数，kernel size=卷积核尺寸，strides=卷积步长，padding = " valid" or "same")`
- LSTM层；`tf.keras.layers.LSTM()`

### compile配置神经网络的训练方法

告知训练时选择的优化器、损失函数和评测指标

model.compile(optimizer = 优化器, loss = 损失函数, metrics = ["准确率"] )

优化器可以是以字符串形式给出的优化器名字

Optimizer（优化器）可选:

- `'sgd'` or `tf.keras optimizers.SGD (lr=学习率,momentum=动量参数)`
- `'adagrad'` or `tf.keras.optimizers.Adagrad (lr=学习率)`
- '`adadelta'` or `tf.keras.optimizers.Adadelta (lr=学习率)`
- `'adam'` or `tf.keras.optimizers.Adam (lr=学习率，beta_ 1=0.9, beta_ 2=0.999)`

loss是（损失函数）可选:

- `'mse'` or `tf.keras losses MeanSquaredError()`
- `'sparse_ categorical_crossentropy` or `tf.keras.losses.SparseCategoricalCrossentropy(from_logits =False)`
  - `from_logits`参数：有些神经网络的输出是经过了softmax等函数的概率分布，有些则不经概率分布直接输出，`from_logits`参数是在询问是否是原始输出，即没有经概率分布的输出。
  - 如果神经网络预测结果输出前经过了概率分布，这里是False
  - 如果神经网络预测结果输出前没有经过了概率分布，直接输出，这里是True

Metrics(评测指标)可选:

`'accuracy'` : y_ 和y都是数值，如y_=[1] y=[1]

`'categorical_accuracy'` : y_ 和y都是独热码(概率分布)，如y_ =[0,1,0] y=[0 256.0.695,0.048]

`'sparse_ categorical_accuracy'` : y_ 是数值，y是独热码(概率分布)，如y_ =[1] y=[0 256,0.695,0.048]

### fit()执行训练过程

model.fit (训练集的输入特征，训练集的标签，
batch_size= ，epochs=,
validation_data=(测试集的输入特征，测试集的标签),
validation_split=从训练集划分多少比例给测试集，
validation_freq =多少次epoch测试一次)

- `batch_ size`：每次喂入神经网络的样本数，推荐个数为：2^n
- `epochs`：要迭代多少次数据集
- `validation_data`和`validation_split`二选一
- `validation_freq`：每多少次epoch迭代使用测试集验证一次结果

### model.summary()打印和统计

`summary()`可以打印出网络的结构和参数统计

### 鸢尾花示例

```python
import tensorflow as tf
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

#鸢尾花分类神经网络，是四输入三输出的一层神经网络，参数12个w和3个b，共计15个参数，这一层是Dense全连接。
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3,activation="softmax",kernel_regularizer=tf.keras.regularizers.l2())
])
```

```python
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=100,validation_split=0.2,validation_freq=20)

model.summary()
```

<!-- #region -->
## 搭建网络class

Sequential搭建神经网络的方法，用Sequential可以搭建出上层输出就是下层输入的顺序网络结构,但是无法写出一些带有跳连的非顺序网络结构。这个时候我们可以选择用类class搭建神经网络结构。

六步法搭建神经网络

1. import
2. train，test
3. class MyModel(Model) model=MyMode
4. model.compile
5. model.fit
6. model.summary

class MyModel(Model):
    # 需要继承Model
	def __init__ (self):
		super(MyModel, self).__init__()
		# 定义网络结构块,super继承要与类名一致
	def cal(self, x):
	# 调用网络结构块，实现前向传播
		return y
model = MyModel()


- `__init__()`:定义所需网络结构块
- call( )：写出前向传播

代码示例
<!-- #endregion -->

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

class IrisModel(Model):
    def __init__(self):
        super(IrisModel,self).__init__()
        self.d1=Dense(3,activation="softmax",kernel_regularizer=tf.keras.regularizers.l2())
        
    def call(self,x):
        y = self.d1(x)
        return y
```

```python
model = IrisModel()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=100,validation_split=0.2,validation_freq=20)

model.summary()
```

## MNIST数据集

提供6万张`28*28`像素点的0~9手写数字图片和标签，用于训练。
提供1万张`28*28`像素点的0~9手写数字图片和标签，用于测试。

代码示例：

```python
import tensorflow as tf
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 可视化训练集输入特征的第一个元素
plt.imshow(x_train[0], cmap='gray')  # 绘制灰度图
plt.show()

# 打印出训练集输入特征的第一个元素
print("x_train[0]:\n", x_train[0])
# 打印出训练集标签的第一个元素
print("y_train[0]:\n", y_train[0])

# 打印出整个训练集输入特征形状
print("x_train.shape:\n", x_train.shape)
# 打印出整个训练集标签的形状
print("y_train.shape:\n", y_train.shape)
# 打印出整个测试集输入特征的形状
print("x_test.shape:\n", x_test.shape)
# 打印出整个测试集标签的形状
print("y_test.shape:\n", y_test.shape)

```

### Sequential实现手写数字识别训练

```python
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train[:2],"\n",y_train[:5])
# 对输入网络的输入特征进行归一化
# 使原本0到255之间的灰度值，变为0到1之间的数值
# 把输入特征的数值变小更适合神经网络训练
x_train, x_test = x_train / 255.0, x_test / 255.0

# 配置模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),   # 输入数据结构拉平
    tf.keras.layers.Dense(128,activation="relu"), # 128个神经元
    tf.keras.layers.Dense(10,activation="softmax") # 10分类的任务，10个输出节点
 ])

model.compile(optimizer="adam",
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)

model.summary()
```

### Class实现手写数字识别训练

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class MnistModel(Model):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128,activation="relu")
        self.d2 = Dense(10,activation="softmax")
    
    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y
model = MnistModel()

model.compile(optimizer="adam",
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             metrics=['sparse_categorical_accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)
model.summary()
```

```python

```
