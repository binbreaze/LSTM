import os
import tensorflow as tf
from series_to_superviser import *
#设置使用两个显卡
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config_gpu)

import numpy as np
import pandas as pd
data = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv',
                   date_parser=lambda date : pd.datetime.strptime(date,'%Y %m %d %H'),
                   parse_dates=[['year','month','day','hour']],index_col=0)

data.drop('No',axis=1,inplace=True)

data = data.ix[24:,:]

data.index.name = 'date'
print(data.columns)
print(data.shape)
data.fillna(axis=0,method='ffill',inplace=True)
data = data.values
timesteps = 4
# print(np.any(np.isnan(data)))
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
data[:,4] = encode.fit_transform(data[:,4])

dataset =  series_to_supervised(data,timesteps-1,1,dropnan=True)
dataset = dataset.ix[:,:-7]
dataset = dataset.reset_index(drop=True)



# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# data = scaler.fit_transform(dataset)
data = np.array(dataset)
print(data.shape)
data = data[:43000,:]
# x = data[:,:-1]
# y = data[:,-1]
train = np.array(np.split(data,430))
# y = np.array(np.split(y,438))
# x = x.reshape(x.shape[0],timesteps,x.shape[1],x.shape[2],1)
# y = y.reshape(y.shape[0],timesteps,y.shape[1])

# data = data.reshape(data.shape[0],timesteps,data.shape[1],data.shape[2],1)
print(train.shape)
x = train[:,:,:-1]
y = train[:,:,-1]
print(x.shape)
print(y.shape)
def to_supervised(data_x,data_y,n_input=timesteps,n_output=timesteps):
    n_start = 0
    train_X , train_y = list(),list()
    for _ in range(len(train)):
        n_end = n_start + n_input
        n_out = n_end + n_output
        if n_out < train.shape[0]:
            # x,y = np.split(train,(train.shape[2]-1,),axis=2)
            x = data_x[n_start:n_end,:,:]
            # print(data_x.shape)
            y = data_y[n_end:n_out,:]
            # print(data_y.shape)
            train_X.append(x)
            train_y.append(y)
        n_start += 1
    return np.array(train_X),np.array(train_y)
train_x ,train_y = to_supervised(x,y)
print(train_x.shape,train_y.shape)
# # #
train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],train_x.shape[3],1)
# # # train_y = train_y.reshape(train_y.shape[0],train_y.shape[1],train_y.shape[2],1,1)
# # print(train_x.shape)
train_x = train_x[:400]
test_x = train_x[400:]
train_y = train_y[:400]
test_y  =  train_y[400:]
# # #
from keras.models import Sequential
from keras.layers import ConvLSTM2D,LSTM,RepeatVector,TimeDistributed,Flatten,Dense,Conv3D
from keras.layers import BatchNormalization
#
model = Sequential()
#
model.add(ConvLSTM2D(filters=20,
                     kernel_size=(3,3),
                     activation='tanh',
                     strides=(1,1),
                     data_format='channels_last',
                     # return_sequences=True,
                     padding='same',
                     input_shape=(train_x.shape[1],train_x.shape[2], train_x.shape[3],1)))

model.add(Flatten())

model.add(RepeatVector(timesteps))

model.add(LSTM(10,activation='relu',return_sequences=True))
# # #
model.add(TimeDistributed(Dense(100)))
#
#
# from keras import losses
#
# # def my_loss(y_true, y_pred):
# #     # y_true = y_true.reshape(y_true.shape[0],1)
# #     # y_pred = y_pred.reshape(y_pred.shape[0],1)
# #     print(y_true.shape)
# #     print(y_pred.shape)
# #     # return np.mean(y_pred)
#
from keras import optimizers
optimizer_adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# sgd = optimizers.SGD(lr=0.01, clipnorm=1)
#
model.compile(loss='mean_squared_error', optimizer=optimizer_adam,metrics=['mae','mape'])
#
model.summary()
history = model.fit(train_x, train_y, epochs=100, batch_size=100,
                    verbose=1,
                    validation_data=(test_x,test_y))
import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='loss')
plt.legend()
plt.show()


model.save('model_convlstm2d_time4_epochs100_batch10_0729.h5')
from keras.models import  load_model
#
model = load_model('model_convlstm2d_time4_epochs100_batch10_0729.h5')

pre = model.predict(test_x[0:1])
print(pre)
