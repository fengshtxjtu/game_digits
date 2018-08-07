
# coding: utf-8

# In[19]:

import keras
from keras.datasets import mnist
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.utils import np_utils
import config.config as config
import config.globalvar as var


batch_size = var.BATCH_SIZE
num_classes = var.NUM_CLASSES
epochs = var.EPOCHS
# # 测试训练集划分
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print type(X_train),type(y_train)
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], var.IMAGE_RESHAPE, var.IMAGE, var.IMAGE).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], var.IMAGE_RESHAPE, var.IMAGE, var.IMAGE).astype('float32')
    input_shape = (var.IMAGE_RESHAPE, var.IMAGE, var.IMAGE)
else:
    X_train = X_train.reshape(X_train.shape[0], var.IMAGE, var.IMAGE, var.IMAGE_RESHAPE).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], var.IMAGE, var.IMAGE, var.IMAGE_RESHAPE).astype('float32')
    input_shape = (var.IMAGE, var.IMAGE, var.IMAGE_RESHAPE)

# X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
# X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


model=Sequential()
model.add(Conv2D(var.FILTER_1,(var.KERNEL_SIZE,var.KERNEL_SIZE),input_shape=(var.IMAGE,var.IMAGE,var.IMAGE_RESHAPE),padding='same',activation='relu'))
model.add(Conv2D(var.FILTER_1,(var.KERNEL_SIZE,var.KERNEL_SIZE),padding='same',activation='relu'))
model.add(MaxPooling2D(strides=var.STRIDES,padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(var.FILTER_2,(var.KERNEL_SIZE,var.KERNEL_SIZE),padding='same',activation='relu'))
model.add(Conv2D(var.FILTER_2,(var.KERNEL_SIZE,var.KERNEL_SIZE),padding='same',activation='relu'))
model.add(MaxPooling2D(strides=var.STRIDES,padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(var.FILTER_3,(var.KERNEL_SIZE,var.KERNEL_SIZE),padding='same',activation='relu'))
model.add(MaxPooling2D(strides=var.STRIDES,padding='same'))
model.add(Flatten())
model.add(Dense(var.FILTER_5,activation="relu"))
model.add(Dropout(var.RATE))
model.add(Dense(var.FILTER_4,activation="relu"))
model.add(Dropout(var.RATE))
model.add(Dense(var.CATEGORY,activation="softmax"))
print(model.summary())
'''
三个参数:损失函数   优化器     指示指标
'''
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

'''
训练模型
'''

model.fit(X_train,y_train,verbose=1,validation_data=(X_test,y_test))

model.save(config.get_config("result","model"))

score=model.evaluate(X_test,y_test)
print('score:',score)

