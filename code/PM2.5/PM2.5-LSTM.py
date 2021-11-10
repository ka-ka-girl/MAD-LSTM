import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#读取数据
data = pd.read_csv('C:/ksy/python-daima/pm2.5论文实验/北京空气_2010.1.1-2014.12.31.csv')
print(data)
print(data.info())


'''
处理缺失数据有两种方法。
1、直接去除比较好，但是会打乱序列，数据不连续了
2、填充
由于前24条我们已经看到全是缺失数据了，所以可以直接去掉前24条。后面的数据用填充的方法。
'''
#首先处理前24个，iloc处理行非常方便，对序列按位置进行索引，下面是取24条之后的数据
data.iloc[24:]
#去掉前24条后的新数据集
data = data.iloc[24:].copy()
#对于剩余的数据仍存在na值，对其进行填充。选择前项填充ffill.
data.fillna(method='ffill', inplace=True)
#pm2.5那一列中数据为na的总个数，处理后的数据集na（缺失）个数为0
print(data['pm2.5'].isna().sum())
#查看缺失数据处理后的数据集
print(data.info())

'''
查看此时的数据发现仍有两个问题需要处理
1、cbwd为python对象（object），要将其数值化
2、把时间设置为索引，设置为时间序列，将多列设置为一列。
使用datetime中datetime模块
'''
#使用datetime中datetime模块将多列数据转化为一列时间序列索引
import datetime
#datetime模块中设置以下参数就可以得到一个时间
datetime.datetime(year=2010, month=1, day=2, hour=1)
#应用一个函数lambda将多列数据转化为一列时间序列索引，列名要加引号。
#axis=1按照每一行进行计算
data['time'] = data.apply(lambda x: datetime.datetime(year=x['year'],
                                       month=x['month'],
                                       day=x['day'],
                                       hour=x['hour']),
                          axis=1)
#data数据中去掉'year', 'month', 'day', 'hour' 'No'这几列。inplace=True表示立即生效
#data.drop('No', axis=1, inplace=True)
data.drop(columns=['year', 'month', 'day', 'hour', 'No'], inplace=True)
#将time列设置为索引
data = data.set_index('time')
#统一化参数名称
data.columns = ['pm2.5', 'dew', 'temp', 'press', 'cbwd', 'iws', 'snow', 'rain']

#将cbwd数值化,首先看一下有哪些数值
data.cbwd.unique()
#数值化。使用pd.get_dummies()方法将其多热编码化
data = data.join(pd.get_dummies(data.cbwd))
#cbwd的数据已经被独热编码，所以去掉这一列
del data['cbwd']
print(data.head())
print(data.info())


'''
划分训练数据
'''
#sequence_length要观测的数据，以下观测前面5天的数据。delay预测未来一天的数据，即24
#训练集为前5天的数据，预测值为后一天的数据（5*24个值再间隔24个的值）。
sequence_length = 5*24
delay = 24

#方法1：直接循环采取数据，由于数据重复，会占用大量内存
data_ = []
for i in range(len(data) - sequence_length - delay):
    data_.append(data.iloc[i: i + sequence_length + delay])


#为了方便处理数据，将数据转化为array
data_ = np.array([df.values for df in data_])
print(data_.shape)
#数据乱序，由于data_内存储的数据是一批一批的，所以打乱顺序也不会打乱原本的序列顺序
np.random.shuffle(data_)
print(data_)


#data_中切出训练数据x，[第一维不管，切前面的5*24或-delay，最后一维不管全都要]
x = data_[:, :-delay, :]
#data_中切出目标数据y，[第一维不动，切最后一个值，要第一个值pm2.5的值]
y = data_[:, -1, 0]
print(x.shape)
print(y.shape)

#由于数据已经做了乱序，所以我们直接使用切片，设定一个比例。
#以下是80%作为训练数据，20%为测试数据
split_boundary = int(data_.shape[0] * 0.8)

train_x = x[: split_boundary]
test_x = x[split_boundary:]

train_y = y[: split_boundary]
test_y = y[split_boundary:]

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)


'''
数据标准化目的：
1、为了把不同特征的取值范围压缩在同一个范围内，有利于预测。
2、有利于网络训练。
注意：
1、计算均值mean和方差std时要在训练数据中计算，而不是所有数据
2、不需要对label（y）做标准化
'''
mean = train_x.mean(axis=0) #计算每一列的均值
std = train_x.std(axis=0)   #计算每一列的方差

print(mean.shape)
#标准化
train_x = (train_x - mean)/std
#标准化，测试数据使用训练数据中计算的均值和方差
test_x = (test_x - mean)/std


'''
Model3:使用多层LSTM堆叠提高预测精度,一定要使用return_sequences=True。
此时不是只有一个输出了，而是每一个观测都输出，所以是一系列（return_sequences）的输出。
添加多层会增强其拟合能力。
最后一层的LSTM不需要return_sequences了，只需要输出一个结果（二维数据），
最后再加一层DENSE （处理二维数据）就可以了。
内部具体结构是怎样运算的？
'''
inputs = tf.keras.layers.Input(shape=(train_x.shape[1:]))
a = tf.keras.layers.LSTM(48, return_sequences=True)(inputs)

#x1 = a*0.9 + b*0.05 + c*0.05
#x2 = a*0.05 + b*0.9 + c*0.05
#x3 = a*0.05 + b*0.05 + c*0.9

a1 = tf.keras.layers.LSTM(48, return_sequences=True)(a)


a2 = tf.keras.layers.LSTM(48)(a1)


#x = tf.keras.layers.Dense(32, activation='relu')(x)
predictions = tf.keras.layers.Dense(1)(a2)

model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

'''
这里loss直接规定为了mae,所以loss就是平均绝对误差mean absolute error (mae)
'''
model.compile(optimizer=keras.optimizers.Adam(), loss='mae')
'''
训练技巧：LSTM层的优化和在训练中降低学习速率
通过keras回调类函数keras.callbacks，使用ReduceLROnPlateau类。
参数（监控目标val_loss，条件patience=3，降低的比例factor=0.5，学习速率最低点）
监控val_loss，如果在3个epoch中val_loss没有降低，那么我们就将低学习速率，学习速率乘以一个factor（比例），
学习速率降到最低时就不再降低。
注意：所有的超参数都需要自己去调试。
'''
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)

#与回调函数相匹配的是在训练过程中使用callbacks，它是一个list,可以同时使用多个类，这里使用了一个。
history = model.fit(train_x, train_y,
                    batch_size = 128,
                    epochs=200,
                    validation_data=(test_x, test_y),
                    callbacks=[learning_rate_reduction])
                    #verbose=0)


f = open("out.txt", "w")    # 打开文件以便写入
print("loss ", history.history.get('loss'), "val-loss ", history.history.get('val_loss'),file=f)
f.close  #  关闭文件

print(history.history.get('loss'))
print(history.history.get('val_loss'))
#实验结果发现有一些过拟合。
plt.plot(history.epoch, history.history.get('loss'), 'y', label='Training loss')
plt.plot(history.epoch, history.history.get('val_loss'), 'b', label='Test loss')
plt.legend()
plt.savefig('Loss')
plt.show()

#存储模型，方便使用
model.save('pm2.5_v3.h5')


'''
1、如何评价model：
model.evaluate()：用来评价，参数为输入数据和对应的实际预测值。这里用划分好的test数据。
平运算得到均的loss值或者平均的准确率是多少。
内部参数verbose=0,表示不显示进度条，直接显示结果。
'''
error = model.evaluate(test_x, test_y, verbose=0)
print(error)
#运算得到平均的损失值。
'''
2、model预测使用：
预测单条数据：
预测多条数据：model.predict()：用来预测，
'''
pre_test = model.predict(test_x)  #返回numpy数据
print(pre_test)
'''
2、model预测使用：
预测单条数据：测试对未知数据的预测，预测2015.1.1，23时pm2.5.
预测多条数据：

data_test = data[-120: ]  #取该观测点之前的 120观测数据
#取所有的行，以及第五列之后的所有列
data_test = data_test.iloc[:, 5:]
#对cbwd进行独热编码，转换成哑变量.如果没有出现某一个值（比如cv），此时要添加0
data_test = data_test.join(pd.get_dummies(data_test.cbwd))
#去掉'cbwd'这一列，所以要加axis=1,又要立即生效所以要加inplace=True。
data_test.drop('cbwd', axis=1, inplace=True)
#测试的特征顺序要与我们训练数据的特征顺序保证一致。如果不确定，可以用以下方法设置
data_test.reindex(columns=['pm2.5', 'dew', 'temp', 'press', 'iws', 'snow', 'rain', 'NE', 'NW', 'SE', 'cv'])
#数据归一化，使用训练数据中得到的均值和方差
data_test = (data_test - mean)/std
#取出numpy类型的数据
data_test = data_test.to_numpy()
#将单条数据增加维度，使其维度为3.在第一维扩展故为0.
data_test = np.expand_dims(data_test, 0)
a = model.predict(data_test)  #预测值为2015年1月1日23时pm2.5的值。（1,1是 41，1.2是51）
'''