'''
File: process_annotation_file.py
Project: prepare_annotation
Created Date: 2023-10-19 05:02:38
Author: chenkaixu
-----
Comment:

Readin image, and convert to the format of the annotation file.
like: path, start_frame, end_frame, disease_id, gait_id
 
Have a good code time!
-----
Last Modified: 2023-10-26 05:26:51
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''

import pandas as pd
from typing import Dict
import os, logging
from prepare_annotation.one_gait_cycle import define_gait_cycle

label_to_id_file = 'labels_to_id.csv'
annotation_in_file = 'train.csv'
annotation_out_file = 'train_processed.txt'
rgb_path = 'rgb/'


""" Read in mapping from class label to class ID """
label_to_id_df = pd.read_csv(label_to_id_file)
label_to_id_dict: Dict[str, int] = dict()
for index, row in label_to_id_df.iterrows():
    label_to_id_dict[row['name']] = int(row['id'])


""" Read in original annotations and convert class label to class ID """
annotations = pd.read_csv(annotation_in_file)
annotations_processed = []
for index, row in annotations.iterrows():
    label, video_id, start_time, end_time = row['label'], row['youtube_id'], row['time_start'], row['time_end']
    annotations_processed.append((video_id, label_to_id_dict[label], start_time, end_time))


""" Find out how many rgb frames each video has and finally write to new annotation file """
with open(annotation_out_file, 'w') as f:
    for annotation in annotations_processed:
        video_id, class_id, start_time, end_time = annotation
        video_id = str(video_id) +'_{:06d}_{:06d}'.format(start_time, end_time)
        start_frame = 0
        frame_path = os.path.join(rgb_path, video_id)
        try:
            num_frames = len(os.listdir(frame_path))
            if num_frames == 0:
                print(f'{video_id} - no frames')
                continue
        except FileNotFoundError as e:
            print(e)
            continue

        annotation_string = "{} {} {} {}\n".format(video_id, start_frame, num_frames-1, class_id)
        f.write(annotation_string)

# %% 