# coding = utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from  keras.models import load_model
from keras.layers import LSTM,Dense,TimeDistributed,Dropout
from keras.models import Sequential
from keras import optimizers
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import os
from  matplotlib import pyplot
from    numpy import concatenate
from   series_to_superviser import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv',
                   date_parser=lambda date : pd.datetime.strptime(date,'%Y %m %d %H'),
                   parse_dates=[['year','month','day','hour']],index_col=0)

data.drop('No',axis=1,inplace=True)

data = data.ix[24:,:]
data.index.name = 'date'

values = data.values
dataset = pd.DataFrame(values)

dataset =  dataset.reset_index(drop=True)

dataset = np.array(dataset)
encode = LabelEncoder()
dataset[:,4] = encode.fit_transform(dataset[:,4])
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

label = dataset[:,0]
dataset = np.delete(dataset,0,axis=1)

timesteps = 8
train_x,test_x,train_y,test_y = train_test_split(dataset,label,test_size=0.2)

train_y = train_y.reshape(len(train_y),1)
test_y = test_y.reshape(len(test_y),1)

train_x = train_x.reshape(train_x.shape[0]//timesteps,timesteps,train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0]//timesteps,timesteps,test_x.shape[1])
train_y = train_y.reshape(train_y.shape[0]//timesteps,timesteps,train_y.shape[1])
test_y = test_y.reshape(test_y.shape[0]//timesteps,timesteps,test_y.shape[1])

model  = Sequential()
model.add(LSTM(128,input_shape=(train_x.shape[1],train_x.shape[2]),activation='tanh',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,activation='tanh',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16,activation='tanh',return_sequences=True))
model.add(TimeDistributed(Dense(1,activation='sigmoid')))
optimizer_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model.compile(loss='mse',optimizer=optimizer_adam,metrics=['accuracy'])
history = model.fit(train_x,train_y,batch_size=200,epochs=500,validation_data=(test_x,test_y))
model.save('my_model_adam_no_superviser_0715.h5')

model = load_model('./my_model_adam_no_superviser_0715.h5')

pyplot.plot(history.history['loss'],label='loss')
pyplot.legend()
pyplot.show()

pre = model.predict(test_x)

pre = pre.reshape(pre.shape[0]*timesteps,pre.shape[2])
print(pre.shape)
print(test_x.shape)
test_x = test_x.reshape(test_x.shape[0]*test_x.shape[1],test_x.shape[2])
inv_yhat = concatenate((pre,test_x),axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

print(inv_yhat)

test_y = test_y.reshape(test_y.shape[0]*test_y.shape[1],test_y.shape[2])
inv_y = concatenate((test_y,test_x),axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
inv_y = pd.Series(inv_y)
inv_y = inv_y.replace(0,0.1)
inv_y = np.array(inv_y)
print(inv_y)
bia = np.abs(inv_y-inv_yhat)


bias = np.abs(inv_yhat-inv_y) / inv_y
bias = np.mean(bias)*100

print('平均偏差：{bias}:'.format(bias=bias))

pyplot.plot(inv_y[:100],label='true')
pyplot.plot(bia[:100],label='bia')
pyplot.plot(inv_yhat[:100],label='pre')
pyplot.legend()
pyplot.show()

pyplot.plot(bia,label='bias')
pyplot.legend()
pyplot.show()
print('预测最大值：{num}:'.format(num=np.max(inv_yhat)))
print('真实值最大值：{num}'.format(num=np.max(inv_y)))
#
# test_x = test_x.reshape(test_x.shape[0],test_x.shape[1]*test_x.shape[2])
# test_x = pca.inverse_transform(test_x)
# inv_hbat = concatenate((pre,test_x),axis=1)
# inv_hbat = scaler.inverse_transform(inv_hbat)
# inv_hbat = inv_hbat[:,0]
#
#
# test_y = test_y.reshape(len(test_y),1)
# inv_y = concatenate((test_y,test_x),axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# inv_y = pd.Series(inv_y)
# inv_y = inv_y.replace(0,0.1)
# inv_y = np.array(inv_y)
# print(inv_y)
#
# bias = np.abs(inv_hbat-inv_y) / inv_y
# bia = np.abs(inv_y-inv_hbat)
#
# mean_bias = np.mean(bias)*100
# print('平均偏差：{bias}'.format(bias=mean_bias))
# print('最大偏差：{bias}'.format(bias=np.max(bias)*100))
# print('最小偏差：{bias}'.format(bias=np.min(bias)*100))
# print('真实值最大：{num}'.format(num=np.max(inv_y)))
#
# pyplot.plot(inv_hbat[:100],label='pre')
# pyplot.plot(inv_y[:100],label='true')
# pyplot.legend()
# pyplot.show()
#
# pyplot.plot(bia,label='bia')
# pyplot.legend()
# pyplot.show()
