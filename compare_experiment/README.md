# Comparative Experiments

We compared the proposed method with several classical approaches, including 2D CNN, CNN-LSTM, 3D CNN, Two-stream, and Skeleton-based methods. We reproduced these methods and tested them on our dataset for comparison. Below, we provide detailed documentation of the reproduction and testing processes for each method.

## 2D CNN Action Recognition

We performed action recognition using a 2D CNN without considering temporal information. Here, we used ResNet50 as the backbone and added a fully connected layer at the end for classification. The model was trained using a pre-trained model on ImageNet.

## CNN-LSTM Action Recognition

For this approach, we considered temporal information using CNN-LSTM for action recognition. The CNN part uses ResNet50 as the backbone for feature extraction. The extracted features are then fed into the LSTM to learn temporal information. Finally, a fully connected layer is added to the LSTM output for classification. The CNN part was trained using a pre-trained model on ImageNet.

## 3D CNN Action Recognition

3D CNN is a network structure specifically designed for processing video data. Here, we used a ResNet50 3D CNN network for action recognition. We used a pre-trained model on the Kinetics dataset for training.

## Two-stream Action Recognition

Two-stream is a network structure that combines optical flow and RGB information. Here, we first converted video frames into static images and then trained RGB and optical flow networks separately. The outputs of the two networks were then fused for action recognition.

For the optical flow network, we used the RAFT method for optical flow estimation and used the optical flow data as input to train the optical flow network. For the RGB network, we used ResNet50 as the backbone for feature extraction. The models were trained using pre-trained models on ImageNet.

We trained for 100 epochs without using early stopping. Detailed configurations can be found in `/compare_experiment/skeleton/stgcn/stgcn_joint-keypoint-2d.py`.

## Skeleton-based Action Recognition

Skeleton-based action recognition methods utilize skeleton data for action recognition. Here, we used STGCN for action recognition.

We first used YOLOv8 to extract skeleton data from videos and then input the skeleton data into STGCN for training. We used a pre-trained model on the NTU RGB+D dataset for training.

### Prepare Skeleton Dataset

Since our dataset is in video format, we need to convert the videos into skeleton data to use skeleton-based methods for action recognition. We referred to the extraction method provided on the mmaction2 official website, using YOLOv8 to extract skeleton data and save it in `.pkl` format.

The extraction script can be executed using `/prepare_skeleton_dataset/main_yolov8.py`. This script uses YOLOv8 for video detection and extracts skeleton data, which includes 18 key points, with each key point containing x, y coordinates and confidence score.

You can generate the skeleton data using the following command:

```python
python main_yolov8.py 
```

## References

1. [mmaction2 preparing skeleton data](https://github.com/open-mmlab/mmaction2/tree/90fc8440961987b7fe3ee99109e2c633c4e30158/tools/data/skeleton)