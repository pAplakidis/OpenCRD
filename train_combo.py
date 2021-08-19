#!/usr/bin/env python3
import numpy as np
import cv2
import pims
from tqdm import trange
from os import listdir
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

from helpers import *
from model import *
from project_polylines import *


# DATA FUNCTIONS
# TODO: return labels for cr detection as well
# get training data from path
def get_data(video_path, log_path, annotations_path):
  # get video frames
  frames = pims.Video(video_path, format="mp4")
  
  with open(log_path, 'r') as log_file:
    labels = log_file.read().split("\n")[:-1]
    log_file.close()
  
  # get road edges data
  annotations = extract_polylines(annotations_path)
  annotations = extract_frame_lines(annotations)
  annotations = convert_annotations((annot_W,annot_H), (W,H), annotations)

  return frames, labels, annotations

def train(frames, labels, annotations, model):
  loss_function = ComboLoss(2, model)
  optim = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-07)
  
  # TODO: we are dealing with 2 tasks, so accuracies might be different
  losses, accuracies = [], []
  epochs = 5
  BS = 32
  
  # TODO: complete this script (check for memory usage!!!)
  for epoch in range(epochs):
    for i in (t := trange(0, len(frames)-BS, BS)):
      pass
  
  return model

if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"
  video_files = []
  log_files = []
  annot_files = []

  for f in listdir(base_dir):
    if f.endswith(".mp4"):
      video_files.append(f)
    elif f.endswith(".xml"):
      annot_files.append(f)
    elif f.endswith(".txt"):
      log_files.append(f)
  video_files, log_files, annot_files = sorted(video_files), sorted(log_files), sorted(annot_files)

  video_files = video_files[:3] # TODO: this is a temp hack, need to get all videos' annotations
  log_files = log_files[:len(video_files)]

  print(video_files)
  print(log_files)
  print(annot_files)
  assert len(video_files) == len(log_files)
  assert len(video_files) == len(annot_files)
  model = ComboModel()

  for i in (t := trange(len(video_files))):
    t.set_description("Loading from files: %s, %s, %s" % ((base_dir+video_files[i], base_dir+log_files[i], base_dir+annot_files[i])))
    frames, labels, annotations = get_data(base_dir+video_files[i], base_dir+log_files[i], base_dir+annot_files[i])
    frames = conv_frames(frames)
    if i == 0:
      all_frames = frames
      all_labels = labels
      all_annotations = annotations
    else:
      all_frames = np.concatenate((all_frames, frames), axis=0)
      all_labels = np.concatenate((all_labels, labels), axis=0)
      all_annotations = np.concatenate((all_annotations, annotations), axis=0)
      
  frames, labels = [], []  # free up some memory
  print("[+] Training model ...")
  model = train(all_frames, all_labels, all_annotations, model)
  print("[+] Done training!")

  # TODO: save model

