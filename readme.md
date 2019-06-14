# 第三次作业
## 要求
1、基于RNN实现文本分类任务，数据使用搜狐新闻数据(SogouCS, 网址：http://www.sogou.com/labs/resource/cs.php)。任务重点在于搭建并训练RNN网络来提取特征，最后通过一个全连接层实现分类目标。
可以参考https://zhuanlan.zhihu.com/p/26729228

2、基于CIFAR-10数据集（https://www.cs.toronto.edu/~kriz/cifar.html）使用CNN完成图像分类任务。

3、基于MNIST数据集（http://yann.lecun.com/exdb/mnist/）使用GAN实现手写图像生成的任务。

---
## 1. RNN实现文本分类任务
### 数据集简介
来自搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据，提供URL和正文信息。
后由助教帮忙将数据转换为一个两列的数据集，第一列为label新闻分类，第二列为文本。

### 数据预处理
数据预处理主要将字符串类型的类别转换成数值类型，再将文本进行分词操作。
```
#读取数据
train_data = pd.read_csv('sohu.csv')
#将字符串类型的类别转换成数值类型
train_data['label']=train_data['label'].map({'pic': 0, 'news': 1, 'sports' : 2, 'business' : 3, 'caipiao' : 4, 'yule' : 5, 'mil' : 6, 'cul' : 7})
for i in train_data['text']:
    #去除文本中的标点符号
    i = sub("[\s+\·<>?《》“”.\!【】\/_,$%^*(：\]\[\-:;+\\']+|[+——！，。？、~@#￥%……&*（）]+","",i)
    #分词
    i = jieba.cut(i, cut_all = True)
train_data['text'] = train_data['text'].apply(lambda x:' '.join(jieba.cut(x)))
```
处理结果如下
![sohu预处理结果](https://img-blog.csdnimg.cn/20190613153200588.png)
### 提取特征
提取特征参照了https://zhuanlan.zhihu.com/p/26729228
使用 keras 的 Tokenizer 来实现，将新闻文档处理成单词索引序列。

```
#提取特征
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['text'])
list_tokenized_train = tokenizer.texts_to_sequences(train_data['text'])
x = pad_sequences(list_tokenized_train, maxlen=100)
y = train_data['label']
```

### 搭建模型
利用序贯模型，添加将文本处理成向量的 embedding 层，这样每个新闻文档被处理成一个 5000 x 128 的二维向量。这里利用了双向RNN处理序列数据，池化层来缩小向量长度，最后全连接层将向量长度收缩。
```
#模型搭建
from keras.models import Model, Sequential
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten, Bidirectional, GlobalMaxPool1D
embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### 训练模型
设置batch_size=50, epochs=5
```
model.fit(x_train, y_train, batch_size=50, epochs=5, validation_split=0.2)
```
结果如下
![RNN训练结果](https://img-blog.csdnimg.cn/20190613155903332.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FudGhvbnlNMDg=,size_16,color_FFFFFF,t_70)
评估
![在这里插入图片描述](https://img-blog.csdnimg.cn/201906131559522.png)
### 误差分析
新闻数量分布不均，没有处理成one hot向量

---
## 2.CIFAR-10数据集图像分类
### 数据集简介
该数据集共有60000张彩色图像，图像为32*32，供10类，每类6000张图，50000张用于训练，构成5个训练批，每一批10000张图；10000张用于测试。测试批的数据取自10类中的每一类，每一类随机取1000张。训练批为剩下的随机抽取。

### 数据预处理
对图像进行归一化，将标签变为one hot向量
```
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
```
### 搭建模型
同样使用序贯模型，CNN为两层卷积层和池化层
```
model = Sequential()
#卷积层和池化层
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32, 32,3), activation='relu', padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
#卷积层和池化层
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2))) 
#建立神经网络
model.add(Flatten()) 
model.add(Dropout(rate=0.25))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax')) 
```
### 训练模型
10批，每批大小为128
```
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
training=model.fit(x_img_train_normalize, y_label_train_OneHot,validation_split=0.2,epochs=10, batch_size=128, verbose=1) 
```
### 模型评估
```
scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot, verbose=0)
```
结果为0.8069150501251221

---

## 3.MNIST数据集 GAN实现手写图像生成
### 数据集简介
MNIST是一个手写数字图像的数据集，每幅图像都由一个整数标记。数据集包含一个有6万个样例的训练集和一个有1万个样例的测试集。训练集用于让算法学习如何准确地预测出图像的整数标签，而测试集则用于检查已训练网络的预测有多准确。

### GAN原理
GAN的由两部分组成： 
判别器(Discriminator) + 生成器(Generator)
生成器主要用来学习真实图像分布从而让自身生成的图像更加真实，以骗过判别器。判别器则需要对接收的图片进行真假判别。
在整个过程中，生成器努力地让生成的图像更加真实，而判别器则努力地去识别出图像的真假，这个过程相当于一个二人博弈，随着时间的推移，生成器和判别器在不断地进行对抗，最终两个网络达到了一个动态均衡：生成器生成的图像接近于真实图像分布，而判别器识别不出真假图像，对于给定图像的预测为真的概率基本接近0.5（相当于随机猜测类别）

### 读取数据
因为数据集已经处理完毕，直接用tensorflow自带的input函数就可以读取mnist_data
```
mnist = input_data.read_data_sets("mnist_data")
```
### 输入函数
```
# 模型输入
def get_inputs(real_size, noise_size):
#实际图像和随机噪声
    real_img = tf.placeholder(tf.float32, [None, real_size], name="real_img")
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name="noise_img")
    return real_img, noise_img
```

### 生成器
 noise_img：随机噪声输入
 out_dim,：输出的大小，例如在MNIS中为MNIST
 n_units：中间隐藏层的单元个数
 reuse：是否重复使用
 alpha：LeakyReLU的参数

```
#生成器
def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse=reuse):
        hidden1 = tf.layers.dense(noise_img, n_units)  # 全连接层
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)
        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)
        return logits, outputs
```

### 判别器
img：输入的图像
n_units：中间隐藏层的单元个数
reuse：是否重复使用
alpha：LeakyReLU的参数
```
#判别器
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs
```
### 模型搭建
判别器的loss为 d_loss = d_loss_real + d_loss_fake, d_loss_real和d_loss_fake用的是二元交叉熵
对于真实图像，给定的label全为1，但是作者认为1这个目标有点难，为了让网络更容易训练，把1降低为0.9 
对于假图像，给定的label全为0
生成器的loss为 g_loss，生成器希望欺骗判别器，因此给定的label全为1
```
#超参
img_size = mnist.train.images[0].shape[0]
noise_size = 100
# 隐层神经元个数
g_units = 128
d_units = 128
alpha = 0.01 # Leak factor
learning_rate = 0.001
smooth = 0.1

#搭建模型
tf.reset_default_graph()
real_img, noise_img = get_inputs(img_size, noise_size)
g_logits, g_outputs = get_generator(noise_img, g_units, img_size)
#判别器
d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)
#计算loss
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_real, labels=tf.ones_like(d_logits_real)
) * (1 - smooth))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)
))
d_loss = tf.add(d_loss_real, d_loss_fake)
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)
) * (1 - smooth))
# 获取相应要训练的变量
train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
```

### 训练函数和可视化
```
#训练参数
epochs = 5000
samples = []
n_sample = 10
losses = []

with tf.Session() as sess:
#    初始化参数
    tf.global_variables_initializer().run()
    for e in range(epochs):
        batch_images = samples[e] * 2 -1
        batch_noise = np.random.uniform(-1, 1, size=noise_size)
 
        _ = sess.run(d_train_opt, feed_dict={real_img:[batch_images], noise_img:[batch_noise]})
        _ = sess.run(g_train_opt, feed_dict={noise_img:[batch_noise]})
 
    sample_noise = np.random.uniform(-1, 1, size=noise_size)
    g_logit, g_output = sess.run(get_generator(noise_img, g_units, img_size,
                                         reuse=True), feed_dict={
        noise_img:[sample_noise]
    })
    print(g_logit.size)
    g_output = (g_output+1)/2
    plt.imshow(g_output.reshape([28, 28]), cmap='Greys_r')
    plt.show()
```
结果
![GAN MNIST](https://img-blog.csdnimg.cn/20190614101321882.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FudGhvbnlNMDg=,size_16,color_FFFFFF,t_70)
很明显能分别出是2
