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

# TODO: desire for each frame is needed! (more labeling)
def get_data(video_path, annotations_path):
  frames = pims.Video(video_path, format="mp4")

  path = extract_polylines(annotations_path)
  path = extract_frame_lines(path)
  path = convert_annotations((path_W,path_H), (W,H), path)

  return frames, path

def train():
  pass


if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"

  video_files = []
  path_files = []
  for f in listdir(base_dir):
    if f.endswith(".mp4"):
      video_files.append(f)
    elif f.endswith("path.xml"):
      path_files.append(f)
  video_files, path_files = sorted(video_files), sorted(path_files)

  video_files = video_files[:3] # TODO: this is a temp hack, need to get all videos' annotations
  print(video_files)
  print(path_files)

  assert len(video_files) == len(path_files), "Number of video files != number of annotation files"

  # TODO: define the model in model.py
  model = PathPlanner()

  # TODO: add desires in the training dataset as well
  for i in trange(0, len(video_files)):
    print("[~] Loading from files: %s , %s" % (base_dir+video_files[i], base_dir+path_files[i]))
    frames, path = get_data(base_dir+video_files[i], base_dir+path_files[i])
    frames = conv_frames(frames)
    if i == 0:
      all_frames = frames
      all_paths= path
    else:
      all_frames = np.concatenate((all_frames, frames), axis=0)
      all_paths = np.concatenate((all_paths, path), axis=0)

  #frames, path = [], [] # free up memory
  print("[+] Training model ...")
  #model = train(all_frames, all_paths[:-1], model)
  print("[+] Trained model on all data files")

