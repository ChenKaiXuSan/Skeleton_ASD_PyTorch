# 对比试验

## Prepare Skeleton dataset

由于我们的数据集是视频格式，如果想要使用基于skeleton的方法来进行动作识别，我们需要先将视频转换为skeleton数据。
这里我们参考了mmaction2官网的提取方式，使用Yolov8来提取skeleton数据并保存成为.pkl格式文件。

具体的提取脚本在`/prepare_skeleton_dataset/main_yolov8.py`中执行。
这里使用yolov8来对视频进行检测，然后提取出skeleton数据。包含18个关键点，每个关键点包含x,y坐标和置信度。

你可以使用如下命令来生成skeleton数据：

```python3
python main_yolov8.py 
```

## STGCN

## References

1. [mmaction2 preparing skeleton data](https://github.com/open-mmlab/mmaction2/tree/90fc8440961987b7fe3ee99109e2c633c4e30158/tools/data/skeleton)
