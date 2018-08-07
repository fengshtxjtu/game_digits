# -*- coding: utf-8 -*-
import ConfigParser


# 定义方法的方式返回value
def get_config(section,key):

    config_parameters = ConfigParser.ConfigParser()
    config_parameters.read("../config/parameter.config")
    # 判断section和key是否存在
    sections = config_parameters.sections()
    if section not in sections:
        print "This section doesn't exist！"

    return config_parameters.get(section,key)