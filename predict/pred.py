# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
import cv2
import config.config as config
import config.globalvar as var

img1 = cv2.imread(config.get_config("pred_image_path","pred_image"),0)

cv2.imshow("img1", img1)
img1 = cv2.resize(img1, (var.IMAGE,var.IMAGE))#通过调整图像大小来对图像进行预处理

img1 = np.reshape(img1,var.IMAGE_RESHAPE).astype('float32')
img1 = abs(img1-255.0)
# print img1.shape

# 载入模型
model = load_model(config.get_config("result","model"))
preds = model.predict(img1,verbose=0)
print preds[0]
labels = range(var.CATEGORY)
name = labels[np.argmax(preds)]
print name
cv2.waitKey(0)