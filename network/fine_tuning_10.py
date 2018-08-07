# coding: utf-8

import cv2
import os
import numpy as np
from keras.models import load_model
import config.config as config
import config.globalvar as var

#数据处理
X = []
Y = []
path=config.get_config("1679","path")
def getImg (path):
    parents = os.listdir(path)
    for parent in parents:
        # print parent
        child = os.path.join(path, parent)
        # print(child)
        if os.path.isdir(child):
            Y.append(parent)
            getImg(child)
            # print(child)
        else:
            # print child
            img = cv2.imread(child, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(var.IMAGE,var.IMAGE))
            X.append(img)

getImg(path)
x_train = np.array(X)
num = x_train.__len__()
x_train = np.reshape(x_train, (num, var.IMAGE, var.IMAGE, var.IMAGE_RESHAPE)).astype('float32')
x_train = abs(x_train - 255.0)
print x_train.shape
y_train = []
for i in Y:
    for j in range(150):
        y_train.append(i)
y_train = np.array(y_train)
print y_train
x_test = x_train[101:]
y_test = y_train[101:]
# 模型处理
model = load_model(config.get_config("result","model_1"))
for i, layer in enumerate(model.layers):
   print(i, layer.name,layer.output_shape)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(x_train,y_train,epochs=var.EPOCHS,batch_size=var.BATCH_SIZE,verbose=var.VERBOSE,validation_data=(x_test,y_test))

model.save(config.get_config("result","model_3"))

score=model.evaluate(x_train,y_train)
print('score:',score)