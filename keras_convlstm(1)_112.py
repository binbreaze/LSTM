import os
import tensorflow as tf
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

data = data.ix[24+57:,:]
data.index.name = 'date'
print(data.columns)
print(data.shape)
data = data.dropna()
print(data.isnull().sum())
data = data.values
print(data.shape)
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
data[:,4] = encode.fit_transform(data[:,4])
#
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# data = scaler.fit_transform(data)

data = np.array(np.split(data,417))

timesteps = 4


def to_supervised(train,n_input=timesteps,n_output=1):
    n_start = 0
    train_X , train_y = list(),list()
    for _ in range(len(train)):
        n_end = n_start + n_input
        n_out = n_end + n_output
        if n_out < train.shape[0]:
            x = train[n_start:n_end,:,:]
            y = train[n_end:n_out,:,0]
            train_X.append(x)
            train_y.append(y)
        n_start += 1
    return np.array(train_X),np.array(train_y)
train_x ,train_y = to_supervised(data)
print(train_x.shape,train_y.shape)
train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2] // 10, train_x.shape[2] // 10, train_x.shape[3])
train_y = train_y.reshape(train_y.shape[0],train_y.shape[2] // 10, train_y.shape[2] // 10, 1)

# train_x = train_x[:100]
# test_x = train_x[100:150]
# train_y = train_y[:100]
# test_y  =  train_y[100:150]

train_x = train_x[:400]
test_x = train_x[400:]
train_y = train_y[:400]
test_y  =  train_y[400:]


from keras.models import Sequential
from keras.layers import ConvLSTM2D,LSTM,RepeatVector,TimeDistributed,Flatten,Dense,Conv3D,BatchNormalization


model = Sequential()

model.add(ConvLSTM2D(filters=12,
                     kernel_size=(3,3),
                     activation='relu',
                     strides=(1,1),
                     data_format='channels_last',
                     return_sequences=True,
                     padding='same',
                     input_shape=(train_x.shape[1],train_x.shape[2], train_x.shape[3],8)))

# model.add(Flatten())
model.add(BatchNormalization())
from keras.layers import Reshape,Conv2D

model.add(ConvLSTM2D(filters=12,
                     kernel_size=(3,3),
                     activation='relu',
                     strides=(1,1),
                     data_format='channels_last',
                     return_sequences=False,
                     padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(1,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same'))
# model.add(RepeatVector(4))

# model.add(LSTM(100,activation='relu',return_sequences=True))

# model.add(TimeDistributed(Dense(1,activation='relu')))


from keras import losses
import keras.backend as K

# def my_loss(y_true, y_pred):
#     # y_true = y_true.reshape(y_true.shape[0],1)
#     # y_pred = y_pred.reshape(y_pred.shape[0],1)
#     print(y_true.shape)
#     print(y_pred.shape)
#     # return np.mean(y_pred)

from keras import optimizers
optimizer_adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.compile(loss='mae', optimizer=optimizer_adam,metrics=['mae'])

model.summary()
model.fit(train_x, train_y, epochs=50, batch_size=100, verbose=1,validation_data=(test_x,test_y))
model.save('model_convlstm2d.h5')
from keras.models import  load_model

model = load_model('model_convlstm2d.h5')

pre = model.predict(test_x)
print(np.array(pre).shape)