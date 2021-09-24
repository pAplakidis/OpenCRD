#!/usr/bin/env python
import numpy as np
import cv2
import pims
from os import listdir
from tqdm import trange
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

from helpers import *
from model import *
from polylines import *

def get_data(video_path, annotations_path):
  frames = pims.Video(video_path, format="mp4")

  path = extract_polylines(annotations_path)
  path = extract_frame_lines(path)
  path = convert_annotations((annot_W,annot_H), (W,H), path)

  return frames, path


if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"

  video_files = []
  annot_files = []
  for f in listdir(base_dir):
    if f.endswith(".mp4"):
      video_files.append(f)
    elif f.endswith("path.xml"):
      annot_files.append(f)
  video_files, annot_files = sorted(video_files), sorted(annot_files)

  video_files = video_files[:2] # TODO: this is a temp hack, need to get all videos' annotations
  print(video_files)
  print(annot_files)

  assert len(video_files) == len(annot_files), "Number of video files != number of annotation files"




