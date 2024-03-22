How to define the one gait cycle in a video, and prepare the annotation file for the training.
===

``` mermaid
graph LR
A[video] --> B[images]
B --> C[one gait cycle]
C --> D[annotation file]
D --> E[prepare dataset]
```

📓 The process enternce is `main.py`

## Convert video to image

First from the raw video, we should convert it to images.
In this time, we split the raw video into 4 different view. 

- Front view
- Back view
- Left view
- Right view

## Define the one gait cycle in a video

## Write the annotation file

The format of the annotation file is as follows:

``` json
{
    "image_path": "path/to/image",
    "label": "label"
}
```

# Not use this at this time.