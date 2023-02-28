# 基于tensorflow2.3的垃圾分类系统

教程链接：[【03】手把手教你构建垃圾分类系统-基于tensorflow2.3_dejahu的博客-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/118225446)

课程设计要做一个垃圾分类系统，需要识别可回收垃圾、厨余垃圾、有害垃圾和其他垃圾等四个大类，在网上找到了很多开源的数据集，但是质量参差不齐，而且有坏图的存在，所以我就将这些数据集还有自己爬取的数据一起清洗了一遍，全部保存为了jpg的格式，一共有245个小类和4个大类。模型训练使用的是tensorflow2.3，其中mobilenet的准确率有82%，并使用pyqt5构建了图形化界面。

## 如何获取

代码直接在本地址下载即可

需要模型和数据集的朋友请在csdn下载，链接如下：

[垃圾分类数据集和tf代码-8w张图片245个类.zip-深度学习文档类资源-CSDN下载](https://download.csdn.net/download/ECHOSON/19713816)

## 代码结构

```
images 目录主要是放置一些图片，包括测试的图片和ui界面使用的图片
models 目录下放置训练好的两组模型，分别是cnn模型和mobilenet的模型
results 目录下放置的是训练的训练过程的一些可视化的图，两个txt文件是训练过程中的输出，两个图是两个模型训练过程中训练集和验证集准确率和loss变化曲线
utils 是主要是我测试的时候写的一些文件，对这个项目没有实际的用途
mainwindow.py 是界面文件，主要是利用pyqt5完成的界面，通过上传图片可以对图片种类进行预测
testmodel.py 是测试文件，主要是用于测试两组模型在验证集上的准确率，这个信息你从results的txt的输出中也能获取
train_cnn.py 是训练cnn模型的代码
train_mobilenet.py 是训练mobilenet模型的代码
```

## 效果
![image-20210618133633509](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210618133633509.png)

## 如何运行

不熟悉环境配置的朋友可以看这篇博客，里面有详细的教程：

[手把手教你用tensorflow2.3训练自己的分类数据集_dejahu的博客-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/117964477?spm=1001.2014.3001.5502)