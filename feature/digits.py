# -*- coding: utf-8 -*-
# USAGE
# python recognize_digits.py

from imutils import contours
import imutils
import cv2
from PIL import Image
import config.config as config
import config.globalvar as var

# load the example image加载实例照片
image = cv2.imread(config.get_config("single_image_path","save_path"))
# print image

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map，
image = imutils.resize(image, height=var.IMAGE_HEIGHT)#通过调整图像大小来对图像进行预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#将其转换为灰度级，
blurred = cv2.GaussianBlur(gray, var.KERNEL, 0)#应用高斯模糊与  5×5 内核降低高频噪声。
edged = cv2.Canny(blurred, 50, 200, 255)#计算边缘图
cv2.imshow("Edged1", edged)


# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image阈值变形图像，然后应用一系列形态学操作来清除阈值图像
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow('thresh',thresh)

# find contours in the thresholded image, then initialize the
# digit contours lists在图像中找到轮廓，然后初始化数字轮廓列表
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour计算轮廓的边界框
	(x, y, w, h) = cv2.boundingRect(c)
	# if the contour is sufficiently large, it must be a digit如果轮廓足够大，它必须是数字
	if w >= 10 and (h >= 30 and h <= 40):
		digitCnts.append(c)
	if (w >=7 and w <= 12) and (h >= 10 and h <= 12):
		cv2.rectangle(thresh, (x, y), (x + w, y + h), var.BOERDER_COLOR, 1)  # 描绘边框
		point = x


# sort the contours from left-to-right, then initialize the
# actual digits themselves从左到右排列轮廓
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]

i = 1
#在digitCnts中循环轮廓，并绘制图像上的边界框
for dc in digitCnts:
	(x, y, w, h) = cv2.boundingRect(dc)
	# print (x, y, w, h)
	# cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 0, 255), 1)#描绘边框
	# 截取数字并保存
	thresh_img = Image.fromarray(thresh)
	cropImg = thresh_img.crop((x, y, x + w, y + h))
	if x < point:
		path = config.get_config("small_image_path","before") + str(i) + '.png'
		cropImg.save(path)
	else:
		path = config.get_config("small_image_path","after") + str(i) + '.png'
		cropImg.save(path)
	i = i+1

#在digitCnts中循环轮廓，并绘制图像上的边界框
for dc in digitCnts:
	(x, y, w, h) = cv2.boundingRect(dc)
	# print (x, y, w, h)
	cv2.rectangle(thresh, (x, y), (x + w, y + h), var.BOERDER_COLOR, 1)#描绘边框

cv2.imshow('thresh2', thresh)

cv2.waitKey(0)

#识别数字——通过神经网络