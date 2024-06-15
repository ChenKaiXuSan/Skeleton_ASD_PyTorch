# 对比试验

我们比较了提出的方法和一些经典的方法，包括2D CNN、CNN-LSTM、3D CNN、Two-stream、Skeleton-based等方法。
我们分别复现了这些方法，并在我们的数据集上进行了测试，以进行对比。
下面我们详细记录了每个方法的复现和测试过程。

## 2D CNN Action Recognition

我们在不考虑时间信息的情况下，使用2D CNN来进行动作识别。
在这里我们使用Restnet50作为backbone，然后在最后加上一个全连接层来进行分类。
使用了在ImageNet上预训练的模型来进行训练。

## CNN-LSTM Action Recognition

我们在考虑时间信息的情况下，使用CNN-LSTM来进行动作识别。
其中CNN部分使用Restnet50作为backbone，进行特征提取。
然后将提取的特征输入到LSTM中，来学习时间信息。
最后在LSTM的输出上加上一个全连接层来进行分类。
CNN部分使用了在ImageNet上预训练的模型来进行训练。

## 3D CNN Action Recognition

3D CNN是一种专门用来处理视频数据的网络结构。
在这里我们使用了Resnet50 3DCNN网络来进行动作识别。
我们使用了在Kinetics上预训练的模型来进行训练。

## Two-stream Action Recognition

Two-stream是一种结合了光流信息和RGB信息的网络结构。
在这里我们首先将视频帧转换为静态图像，然后分别训练RGB和光流网络。
最后将两个网络的输出进行融合，来进行动作识别。

对于光流网络，我们使用了RAFT方法来进行光流估计，然后将光流数据作为输入训练光流网络。
对于RGB网络，我们使用了Resnet50作为backbone，进行特征提取。
我们使用在ImageNet上预训练的模型来进行训练。

我们训练100个epoch，没有使用early stopping。详细配置可以在`/compare_experiment/skeleton/stgcn/stgcn_joint-keypoint-2d.py`中查看。

## Skeleton-based Action Recognition

Skeleton-based是一种基于骨架数据的动作识别方法。
在这里我们使用了STGCN来进行动作识别。

我们首先使用YOLOv8来提取视频中的骨架数据，然后将骨架数据输入到STGCN中进行训练。
这里我们使用了在NTU RGB+D上预训练的模型来进行训练。

### Prepare Skeleton dataset

由于我们的数据集是视频格式，如果想要使用基于skeleton的方法来进行动作识别，我们需要先将视频转换为skeleton数据。
这里我们参考了mmaction2官网的提取方式，使用Yolov8来提取skeleton数据并保存成为.pkl格式文件。

具体的提取脚本在`/prepare_skeleton_dataset/main_yolov8.py`中执行。
这里使用yolov8来对视频进行检测，然后提取出skeleton数据。包含18个关键点，每个关键点包含x,y坐标和置信度。

你可以使用如下命令来生成skeleton数据：

```python3
python main_yolov8.py 
```

## References

1. [mmaction2 preparing skeleton data](https://github.com/open-mmlab/mmaction2/tree/90fc8440961987b7fe3ee99109e2c633c4e30158/tools/data/skeleton)
