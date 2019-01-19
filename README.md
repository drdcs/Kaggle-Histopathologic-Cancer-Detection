# Kaggle-Histopathologic-Cancer-Detection

比赛地址：[Kaggle-Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

模型使用了Resnet50，修改最后几层网络结构重新训练，并且使用了五折交叉验证取平均值来提高精度。其中的一些trick在代码中已经标注。后面考虑修改网络结构，损失函数来进一步提高精度。

--- 

`train_model_seq_resnet.py`这个代码参考了论文[ Squeeze-and-Excitation Networks ](https://arxiv.org/abs/1709.01507)。

SENet的核心思想在于通过网络根据loss去学习特征权重，使得有效的feature map权重大，无效或效果小的feature map权重小的方式训练模型达到更好的结果。当然，SE block嵌在原有的一些分类网络中不可避免地增加了一些参数和计算量，但是在效果面前还是可以接受的。

参考文章：[SENet（Squeeze-and-Excitation Networks）算法笔记](https://blog.csdn.net/u014380165/article/details/78006626)

参考代码：[yoheikikuta/senet-keras](https://github.com/yoheikikuta/senet-keras)
