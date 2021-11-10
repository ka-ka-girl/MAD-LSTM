from scipy.integrate import odeint
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0

def lorenz(w, t, p, r, b):
    x, y, z = w
    return np.array([p*(y-x), x*(r-z)-y, x*y-b*z])

t = np.arange(0, 100, 0.01)
data = odeint(lorenz, (0.0, 1.00, 0.0), t, args=(10.0, 28.0, 3.0))
print(data)
print(data.shape)
'''
划分训练数据
'''
#sequence_length要观测的数据,delay预测未来一天的数据，即24
#训练集为前5天的数据，预测值为后一天的数据（5*24个值再间隔24个的值）。
sequence_length = 200
delay = 1
#方法1：直接循环采取数据，由于数据重复，会占用大量内存
data_ = []
for i in range(len(data) - sequence_length - delay):
    data_.append(data[i: i + sequence_length + delay])


#为了方便处理数据，将数据转化为array
data_ = np.array([df for df in data_])
print(data_.shape)


#数据乱序，由于data_内存储的数据是一批一批的，所以打乱顺序也不会打乱原本的序列顺序
data_sf = data_
np.random.shuffle(data_sf)
print(data_sf)


#data_中切出训练数据x，[第一维不管，切前面的5*24或-delay，最后一维不管全都要]
x = data_sf[:, :-delay, :]
#data_中切出目标数据y，[第一维不动，切最后一个值，要第一个值pm2.5的值]
y = data_sf[:, -1, :]
print(x.shape, y.shape)

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


inputs = tf.keras.layers.Input(shape=(train_x.shape[1:]))
a = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)


#x1 = a*0.9 + b*0.05 + c*0.05
#x2 = a*0.05 + b*0.9 + c*0.05
#x3 = a*0.05 + b*0.05 + c*0.9

a1 = tf.keras.layers.LSTM(32, return_sequences=True)(a)


a2 = tf.keras.layers.LSTM(32)(a1)

#x = tf.keras.layers.Dense(32, activation='relu')(x)
predictions = tf.keras.layers.Dense(3)(a2)

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
                    #batch_size = 128,
                    batch_size = 128,
                    epochs=300,
                    validation_data=(test_x, test_y),
                    callbacks=[learning_rate_reduction])

f = open("out.txt", "w")    # 打开文件以便写入
print("loss ", history.history.get('loss'), "val-loss ", history.history.get('val_loss'),file=f)
f.close  #  关闭文件

print(history.history.get('loss'))
print(history.history.get('val_loss'))

#存储模型，方便使用
model.save('MAD-LSTM-3-32-300_1.h5')

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

#设定预测数据pre_x,预测后200步
pre_x = data_[:, :-delay, :]
real_x = data_[:, -1, :]

future_len = int(data_.shape[0]) - 1000
pre_x = pre_x[future_len:]
real_x = real_x[future_len:]

pre_data = model.predict(pre_x)
print(pre_data)
print(pre_data.shape)
'''

#创建三维坐标系,画出质点的移动轨迹。
fig = plt.figure()
#ax = Axes3D(fig)                     #创建三维坐标系
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2], c='g')   #画出质点的移动轨迹
#ax.plot(pre_data[:, 0], pre_data[:, 1], pre_data[:, 2], c='r')
#ax.plot(tmp['X'], tmp['Y'], tmp['Z'], c='r')
plt.savefig('轨迹图')
plt.show()

#实验结果发现有一些过拟合。
plt.plot(history.epoch, history.history.get('loss'), 'y', label='Train loss')
plt.plot(history.epoch, history.history.get('val_loss'), 'b', label='Test loss')
plt.title("Loss")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss')
plt.show()



