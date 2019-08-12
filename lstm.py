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
from sklearn.model_selection import train_test_split
from  keras.models import load_model
from keras.layers import LSTM,Dense,TimeDistributed,Dropout
from keras.models import Sequential
from keras import optimizers
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler,Normalizer,RobustScaler
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
data.fillna(axis=0,method='ffill',inplace=True)
data = data.values
timesteps = 12
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
data[:,4] = encode.fit_transform(data[:,4])

dataset =  series_to_supervised(data,timesteps-1,1,dropnan=True)

pm = dataset['var1(t)']
dataset = dataset.drop(['var1(t)'],axis=1)
dataset = concat([dataset,pm],axis=1)
dataset =  dataset.drop(['var2(t-11)', 'var3(t-11)', 'var4(t-11)', 'var5(t-11)',
       'var6(t-11)', 'var7(t-11)', 'var8(t-11)'],axis=1)

dataset = dataset.reset_index(drop=True)

print(dataset.shape)

# scaler = MinMaxScaler(feature_range=(0,1))
# scaler = RobustScaler()
scaler = StandardScaler()

dataset = scaler.fit_transform(dataset)


label = dataset[:,-1]
dataset = np.delete(dataset,-1,axis=1)
dataset = dataset.reshape(dataset.shape[0],1,dataset.shape[1])
#
# pca = PCA(n_components=timesteps*7)
# dataset = pca.fit_transform(dataset)
# print(dataset.shape)
# print(label.shape)

# # split_num = 0.2
# # train_x = dataset[:dataset.shape[0]*(1-split_num),:]
# # test_x = dataset[dataset.shape[0]*(1-split_num):,:]
# # train_y = label[:dataset.shape[0]*(1-split_num)]
# # test_y = label[dataset.shape[0]*(1-split_num):]
# train_x = dataset[:30001,:]
# test_x = dataset[30001:,:]
# train_y = label[:30001]
# test_y = label[30001:]


# train_x,test_x,train_y,test_y = train_test_split(dataset,label,test_size=0.2)
train_x = dataset[:30000]
train_y = label[:30000]
val_x = dataset[30000:35000]
val_y = label[30000:35000]
test_x = dataset[35000:]
test_y = label[35000:]
print(dataset.shape)
#
# train_x = train_x.reshape(train_x.shape[0],timesteps-1,train_x.shape[1]//(timesteps-1))
# test_x = test_x.reshape(test_x.shape[0],timesteps-1,test_x.shape[1]//(timesteps-1))
#
model  = Sequential()
# model.add(LSTM(128,input_shape=(train_x.shape[1],train_x.shape[2]),activation='tanh',return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(30,activation='relu',input_shape=(train_x.shape[1],train_x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(8,activation='relu'))
# model.add(LSTM(32,activation='relu',return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(8,activation='relu'))
# model.add(Dense(8,activation='relu'))
model.add(Dense(1))
# optimizer_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

from keras import metrics
model.compile(loss='mse',optimizer='adam',metrics=[metrics.mae,metrics.mape,'accuracy'])
model.summary()
from    keras import callbacks
# callback_list = [
#     callbacks.ReduceLROnPlateau(
#                 monitor = 'val_loss',  # 监控模型的验证损失
#                 factor = 0.1,  # 触发时将学习率除以10
#                 patience = 10
#     )
# ]
history = model.fit(train_x,train_y,batch_size=1000,epochs=100,
                    validation_data=(test_x,test_y),verbose=1)

pyplot.plot(history.history['loss'],label='loss')
pyplot.plot(history.history['val_loss'],label='val_loss')
pyplot.legend()
pyplot.show()

model.save('my_model_adam_time8_units100_std_batch2000_0802.h5')


model = load_model('./my_model_adam_time8_units100_std_batch2000_0802.h5')

pre = model.predict(test_x)

test_x = test_x.reshape(test_x.shape[0],test_x.shape[1]*test_x.shape[2])
# test_x = pca.inverse_transform(test_x)
inv_hbat = concatenate((test_x,pre),axis=1)
inv_hbat = scaler.inverse_transform(inv_hbat)
# pre_test = pd.DataFrame(inv_hbat)
#
# pre_test = pre_test.to_csv('pre_test.csv')

inv_hbat = inv_hbat[:,-1]
print(inv_hbat)


test_y = test_y.reshape(len(test_y),1)
inv_y = concatenate((test_x,test_y),axis=1)
inv_y = scaler.inverse_transform(inv_y)
# y_test = pd.DataFrame(inv_y)
# y_test = y_test.to_csv('y_test.csv')
inv_y = inv_y[:,-1]
print(inv_y)


inv_y = pd.Series(inv_y)
inv_y = inv_y.replace(0,0.1)
inv_y = np.array(inv_y)

bias = np.abs(inv_hbat-inv_y) / inv_y
bia = np.abs(inv_y-inv_hbat)

mean_bias = np.mean(bias)*100

inv_y_mean = np.mean(inv_y)

bias_mean = np.mean(bia)/ inv_y_mean

bia = bia.reshape(len(bia),1)

# bia_5 = len(np.where(bia<5)) / len(bia)
bia = np.array(bia)
bia_10 = bia[np.where(bia<10)]
bia_10_acc = bia_10.shape[0] / bia.shape[0]

print('误差小于10：{num}'.format(num=bia_10_acc*100))

bia_15 = bia[np.where(bia<15)]
bia_15_acc = bia_15.shape[0] / bia.shape[0]
print('误差小于15：{num}'.format(num=bia_15_acc*100))

bia_20 = bia[np.where(bia<20)]
bia_20_acc = bia_20.shape[0] / bia.shape[0]
print('误差小于20：{num}'.format(num=bia_20_acc*100))

bia_30 = bia[np.where(bia<30)]
bia_30_acc = bia_30.shape[0] / bia.shape[0]
print('误差小于30：{num}'.format(num=bia_30_acc*100))


mse = np.mean((inv_y-inv_hbat)**2)
print('mse:{mse}'.format(mse=mse))
rmse = mse**0.5
print('rmse:{rmse}'.format(rmse=rmse))

print('mape：{bias}'.format(bias=mean_bias))
print('mae:{num}'.format(num=np.mean(bia)))
print('最大偏差mae：{bias}'.format(bias=np.max(bia)))
print('最小偏差mae：{bias}'.format(bias=np.min(bia)))
print('真实值最大：{num}'.format(num=np.max(inv_y)))
print('真实值的平均值：{num}'.format(num=np.mean(inv_y)))


pyplot.plot(inv_hbat[:100],label='pre')
pyplot.plot(inv_y[:100],label='true')
pyplot.legend()
pyplot.show()

pyplot.plot(bia,label='bia')
pyplot.legend()
pyplot.show()



