"""
File: labeled_video_dataset.py
Project: dataloader
Created Date: 2023-10-29 11:45:14
Author: chenkaixu
-----
Comment:
This file is used for labeled video dataset.
The step is:
1. load the video path.
2. use pose estimation to get the keypoint.
3. define the gait cycle.
4. split the different gait cycle into list/dict.
5. feed into the dataloader.
 
Have a good code time!
-----
Last Modified: 2023-10-29 12:12:08
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

10-12-2023	Kaixu Chen	这个文件目前可以实现定义，但是在拼装的时候，由于不同患者的不行周期数是不一样的，所以会报错。我决定先把图像周期定义了，然后保存成pt格式会省事一些
                        也就是说这个文件也许可以废弃了。

"""

from __future__ import annotations

import logging, sys
sys.path.append("/workspace/skeleton")

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.utils.data
from torchvision.io import read_video

from pytorchvideo.data.video import Video, VideoPathHandler
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler

from prepare.prepare_gait_dataset.preprocess import Preprocess

logger = logging.getLogger(__name__)


def get_gait_cycle_from_bbox(bbox: torch.Tensor):
    """predict the gait cycle from bbox.
    use the max width bbox as the gait cycle frame index.
    The motivation is that the max width bbox is the most complete bbox in the video.

    Args:
        bbox (torch.Tensor): bbox from the video, shape is [b, t, xywh]

    Returns:
        list: the gait cycle frame index,.
    """

    # find min and max bbox weight
    b, t, xywh = bbox.shape

    bias = 50  # pixel value
    threshold = 10

    min_width = 0
    max_width = float("inf")

    ans = []

    # first process one batch
    for batch in range(b):
        total_width = bbox[batch, :, 2]

        min_width = total_width.min()
        max_width = total_width.max()

        for f in range(t):
            # define the x,y coordinate for left ankel
            x, y, w, h = bbox[batch, f]

            if abs(w - max_width) < bias:
                ans.append(f)

    # filter close values 
    res = [0] 

    for i in range(len(ans)):
        if not res:
            res.append(ans[i])
        else:
            if abs(ans[i] - res[-1]) > threshold:
                res.append(ans[i])

    # return the max width frame index
    return res


class LabeledGaitVideoDataset(torch.utils.data.IterableDataset):
    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
        yolo_config: Dict = None,
    ) -> None:
        super().__init__()

        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder

        # set the preprocess
        self._preprocess = Preprocess(yolo_config)

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)


    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):

            video_index = next(self._video_sampler_iter)
            try:
                video_path, info_dict = self._labeled_videos[video_index]
                video = self.video_path_handler.video_from_path(
                    video_path,
                    decode_audio=self._decode_audio,
                    decoder=self._decoder,
                )
                self._loaded_video_label = (video, info_dict, video_index)


            except Exception as e:
                logger.debug(
                    "Failed to load video with error: {}; trial {}".format(
                        e,
                        i_try,
                    )
                )
                continue

            # * step1:load one video clip from video path
            video_frames, audio, info = read_video(video_path, pts_unit="sec", output_format="TCHW")

            # ! the shape should b, c, t, h, w
            # TCHW > CTHW > BCHW
            frames = video_frames.permute(1,0,2,3).unsqueeze(0)
            # * step2: use preprocess to get the bbox
            # FIXME: notice that if use GPU in dataloader, it will occur err.
            # Maybe need to move the process to __iter__, or set the num worker to 0(now).
            _, label, optical_flow, bbox, mask, pose = self._preprocess(frames, torch.tensor([self._loaded_video_label[1]['label']]), 0)
            # * step3: use bbox to define the gait cycle, and get the gait cycle index
            gait_cycle_index_bbox = get_gait_cycle_from_bbox(bbox)
            # * step4: use gait cycle index to split the video
            frames = []
            for i in range(0, len(gait_cycle_index_bbox)-2, 2):
                frames.append(video_frames[
                    gait_cycle_index_bbox[i]:gait_cycle_index_bbox[i+1]
                ].permute(1,0,2,3)
                )
            
            if self._transform is not None:
                for i in range(len(frames)):
                    frames[i] = self._transform(frames[i])

            audio_samples = None
            sample_dict = {
                "video": torch.stack(frames, dim=0),
                "video_name": video.name,
                "video_index": video_index,
                "gait_index": gait_cycle_index_bbox,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self



def labeled_gait_video_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
    yolo_config: Dict = None,
) -> LabeledGaitVideoDataset:
    """
    A helper function to create ``LabelGaitVideoDataset`` object for self disease datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix

    dataset = LabeledGaitVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
        yolo_config=yolo_config,
    )
    return dataset