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
