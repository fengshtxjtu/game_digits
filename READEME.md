项目应用：游戏人物的识别并框定位置

项目价值：
1、用于百度云游戏人物类游戏识别，根据游戏截图识别对应人物和对应游戏；
2、判定游戏运行状态是否正常，如花屏等异常现象；


实现步骤：
    1、feature
        cutImage：裁剪需要的坐标图片
        digits：将坐标处理成需要的单个数字的图片集
    2、network
        set_network:子搭建网络，用mnist数据集
        fune_tuning:微调网络——目前不用
    3、predict
        pred：单个数字预测，要组合所有数字成为最终结果

目录解析：
	1、config：配置文件

	2、data:
	num

	num_1679

	num_1679_validation

	origin_big_image

	origin_coordinate

	3、result：



