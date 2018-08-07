# -*- coding: utf-8 -*-
from PIL import Image
import os

import config.config as config
import config.globalvar as var
# 单副图像裁剪
def cut_single(dir,path):

    img = Image.open(dir)

    region = var.REGION

    # 裁切图片
    cropImg = img.crop(region)
    # 保存裁切后的图片
    cropImg.save(path)
    return True

#多副图像集体裁剪
def cut_group(dir,save_path):

    region = var.REGION
    i = 0
    for files in os.listdir(dir):
        singlefileName = dir + r"/" + files
        # print singlefileName
        singlefileForm = os.path.splitext(singlefileName)[1][1:]
        if (singlefileForm == 'png'):
            print('loading................ : ', singlefileName)
            img = Image.open(singlefileName)
            cropImg = img.crop(region)
            path = save_path+'jstm_people' + str(i) + '.png'
            print path
            cropImg.save(path)
            i = i + 1



image_dir = config.get_config("group_image_path","image_dir")
save_path = config.get_config("group_image_path","save_path")
cut_group(image_dir,save_path)



image_dir = config.get_config("single_image_path","image_dir")
save_path = config.get_config("single_image_path","save_path")
cut_single(image_dir,save_path)
cut_single(image_dir,save_path)