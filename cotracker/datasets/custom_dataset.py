# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from cotracker.datasets.utils import CoTrackerData

class CustomDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        sequence_length: int,
        step: int = 1,
        crop_size: Optional[Tuple[int, int]] = None,
        split: str = "train",
        min_seq_len: int = 0,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.data_root = os.path.join(data_root, split)  # e.g., /home/imerse/cotracker/data/custom/train
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.step = step
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.video_paths = []
        for file in sorted(os.listdir(self.data_root)):
            if file.endswith(".mp4"):
                video_path = os.path.join(self.data_root, file)
                cap = cv2.VideoCapture(video_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if self.min_seq_len <= length <= self.max_seq_len:
                    self.video_paths.append(video_path)

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames)

    def __getitem__(self, index: int) -> CoTrackerData:
        video_path = self.video_paths[index]
        video = self.read_video(video_path)
        if len(video) < self.sequence_length:
            raise ValueError(f"Video {video_path} too short: {len(video)} frames")
        start = np.random.randint(0, len(video) - self.sequence_length * self.step + 1)
        video = video[start : start + self.sequence_length * self.step : self.step]
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0  # Normalize to [0,1]

        # Dummy tensors (to be replaced by teacher models in training)
        num_points = 128  # Matches --traj_per_sample
        trajectory = torch.zeros((self.sequence_length, num_points, 2), dtype=torch.float32)
        visibility = torch.ones((self.sequence_length, num_points), dtype=torch.bool)
        valid = torch.ones((self.sequence_length, num_points), dtype=torch.bool)  # Add valid tensor

        return CoTrackerData(
            video=video,
            trajectory=trajectory,
            visibility=visibility,
            valid=valid
        )
