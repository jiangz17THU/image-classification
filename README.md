# image-classification
《人工智能基础》第二次大作业，用深度学习网络训练图像分类模型

## 任务描述
    1.本次作业的数据包含30000张图片组成的训练集（train.npy和train.csv）以及5000张图片组成的测试集（test.npy），均可从kaggle网站Data栏目下下载。
  
    2.npy文件可通过numpy.load()函数读取，每个npy文件包含一个N*784的矩阵，N为图片数量。矩阵每行对应一张28*28的图片，同学可在预处理环节自行将每行784维的向量转换成28*28的图像。
  
    3.train.csv文件包含训练集的标签，含image_id和label两列，共30000行，image_id对应矩阵中的行下标，label为该图片的类别标签。
  
    4.在预测环节，需要利用训练好的模型对测试集中的5000张图片进行分类，预测结果应生成submit.csv文件，同样包含image_id和label两列，共5000行，每行对应一张图片的结果。在kaggle网站上提交该文件后会看到自己的分数（指标为catergorization accuracy）及排名,提交文件格式可参考kaggle网站的Data栏目下的samplesummission.csv。


## 实现细节
  1.网络模型选取resnet18，解决梯度消失问题
  
  2.采用GPU加速
  
  3.其他细节见大作业项目报告
