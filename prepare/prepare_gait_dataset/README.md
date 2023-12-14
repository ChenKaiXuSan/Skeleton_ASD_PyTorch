# Preprocess the gait dataset

为了加速计算，应该先处理图片，然后保存成文件，然后再读取文件，这样的话，可以加快计算速度。
也就是说需要想一个文件处理过后应该是什么样子的。

例如,只使用患者步行的前半阶段来进行训练的话，
如果只把一个患者的不行周期分为两部分来看的话，

📓图像增强的操作，从pt文件中读取之后再进行也是可以的

## JSON file

🗒️ Because I found that the .pt file is too large, so I use the json file to store the information of the gait cycle index.

## Format

**This script is used to the segmentation_dataset_512 dataset.**

To define the gait cycle in the video, and save the gait cycle index to json file.

The json file include: (in dict)
``` python
{
    "video_name": the video name,
    "video_path": the video path, relative path from /workspace/skeleton/data/segmentation_dataset_512,
    "frame_count": the raw frames of the video,
    "label": the label of the video,
    "disease": the disease of the video,
    "gait_cycle_index_bbox": the gait cycle index,
    "bbox_none_index": the bbox none index, when use yolo to get the bbox, some frame will not get the bbox.
    "bbox": the bbox, [n, 4] (cxcywh)
}
```

## Pipeline

The python file flow is:

``` mermaid
graph LR
    main.py --> preprocess.py --> yolov8.py
```

The data process pipeline is:

``` mermaid
graph LR
    A[video] --> B[object detection] --> C[BBOX] --> D[.json file]
```

## Usage

``` bash
python main.py
```

The json file will be saved in the indicate folder, like:
`~/dataset/segmentation_dataset_512/json/` folder.