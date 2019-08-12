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
from pandas import DataFrame,concat
data = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv',
                   date_parser=lambda date : pd.datetime.strptime(date,'%Y %m %d %H'),
                   parse_dates=[['year','month','day','hour']],index_col=0)
data.drop('No',axis=1,inplace=True)
data = data.ix[24:,:]
data.index.name = 'date'
data.fillna(axis=0,method='ffill',inplace=True)
data = data.values
timesteps = 12
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
data[:,4] = encode.fit_transform(data[:,4])
print(data.shape)
data = np.array(data)

def to_sequence(data,input_len,output_len):
    start = 0
    X = list()
    y = list()
    for _ in range(len(data)):
        n_end = start + input_len
        n_out = n_end + output_len
        if n_out < len(data):
            X.append(data[start:n_end,1:])
            y.append(data[n_end:n_out,0])
        start += 1
    return np.array(X),np.array(y)
X , y = to_sequence(data,timesteps,timesteps)
print(X.shape,y.shape)
y = y.reshape(y.shape[0],y.shape[1],1)

train_x = X[:35000]
train_y = y[:35000]
valation_x = X[35000:40000]
valation_y = y[35000:40000]
test_x = X[40000:]
test_y = y[40000:]


from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,TimeDistributed,Dropout,RepeatVector

model = Sequential()
model.add(LSTM(30,input_shape=(train_x.shape[1],train_x.shape[2]),activation='relu'))
model.add(Dropout(0.2))
model.add(RepeatVector(timesteps))
model.add(LSTM(30,input_shape=(train_x.shape[1],train_x.shape[2]),activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(10,activation='relu')))
model.add(TimeDistributed(Dense(1)))
from keras import optimizers
optimizer_adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='mse',optimizer='adam',metrics=['mae','mape','acc'])
model.summary()
history = model.fit(train_x,train_y,validation_data=(valation_x,valation_y),batch_size=1000,epochs=100,verbose=1)
model.save('lstm_m2m_time12_x_lr.h5')
import matplotlib.pyplot as plt
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


model = load_model('lstm_m2m_time12_x_lr.h5')
test_y.reshape(test_y.shape[0],test_y.shape[1],1)
pre = model.predict(test_x[10:11])


pre = pre.reshape(1,pre.shape[1])
y = test_y[10:11]
y = y.reshape(1,y.shape[1])

print(pre,y)