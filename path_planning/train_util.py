#!/usr/bin/env python3
import os, sys
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils

from util import *
from model import *

# net input resolution
W = 480
H = 360

# display resolution
d_W = 1920 // 2
d_H = 1080 // 2


# TODO: length of poses and frames do not match (FIX DATA COLLECTOR)
class PathPlannerDataset(Dataset):
    def __init__(self, base_dir):
      super(Dataset, self).__init__()
      self.base_dir = base_dir  # TODO: for now just use one clip
      self.video_path = base_dir + "video.mp4"
      self.poses_path = base_dir + "poses.npy"

      self.poses = np.load(self.poses_path)
      self.local_poses, self.local_path = get_local_poses(self.poses)
      print(self.local_path.shape)

      self.cap = cv2.VideoCapture(self.video_path)

    def __len__(self):
      return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
      #self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx-1)
      self.cap.set(1, idx)
      ret, frame = self.cap.read()
      frame = cv2.resize(frame, (W,H))
      frame = np.moveaxis(frame, -1, 0)
      return frame, self.local_path[idx:LOOKAHEAD+idx]  # TODO: use path for now, later on predict poses


class Trainer:
  def __init__(self):
    pass


if __name__ == "__main__":
  dataset = PathPlannerDataset("../data/sim/8/")
  print(len(dataset))
  img, path = dataset[500]
  print(img.shape)
  print(path.shape)

  disp_img = np.moveaxis(img, 0, -1)
  disp_img = cv2.resize(disp_img, (d_W,d_H))
  print(disp_img.shape)

  draw_path(path, disp_img)
  cv2.imshow("DISPLAY", disp_img)
  cv2.waitKey(0)

  dataset.cap.release()
  cv2.destroyAllWindows()
