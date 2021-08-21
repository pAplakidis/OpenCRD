#!/usr/bin/env python3
import numpy as np
import cv2
import pims
from tqdm import trange
from os import listdir
import xml.etree.ElementTree as ET

import matplotlib
import matplotlib.pyplot as plt

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

  return frames, np.array(labels).astype(np.float), annotations

def train(frames, labels, annotations, model):
  loss_function = ComboLoss(2, model, device)
  optim = torch.optim.Adam(model.parameters(), lr=0.01)
  
  # TODO: we are dealing with 2 tasks, so accuracies might be different
  losses, cr_accuracies, re_accuracies = [], [], []
  epochs = 11
  BS = 32
  
  # TODO: complete this script (check for memory usage!!!)
  print("[+] Training model ...")
  for epoch in range(epochs):
    print("[+] Epoch", epoch+1, "/", epochs)
    epoch_losses = []
    for i in (t := trange(0, len(frames)-BS, BS)):
      X_train = []  # frame
      Y1_train = []  # crossroad bool
      Y2_train = []  # road edge points
      rng = np.random.default_rng()
      samp = rng.choice(len(frames)-1, size=BS, replace=False)  # random sample of size BS

      # prep tensors
      for j in samp:
        frame = frames[j]
        frame = np.moveaxis(frame, -1, 0) # [batch_size, channels, height, width]
        X_train.append(frame)
        Y1_train.append([labels[j]])
        flat_annot = serialize_polylines(annotations[j], model.n_coords, model.n_points, model.max_n_lines)
        Y2_train.append(flat_annot)
      X = torch.tensor(np.array(X_train)).float().to(device)
      Y1 = torch.tensor(np.array(Y1_train)).float().to(device)
      Y2 = torch.tensor(np.array(Y2_train)).float().to(device)

      # forward and backpropagate
      optim.zero_grad()
      out = model(X)

      loss = loss_function(out, Y1, Y2)
      loss.backward() # TODO: this has issues with shapes ([32, 128], [32, 256]), due to device incompatibility in loss!!! [https://blog.csdn.net/andyL_05/article/details/107952479]
      optim.step()

      # TODO: print accuracies as well
      # print stats
      loss = loss.item()
      epoch_losses.append(loss)
      t.set_description("loss %.2f" % (loss))

    losses.append(np.array(epoch_losses).mean())

  # plot losses
  print("[+] Done training!")
  plt.ylim(-1e+8, 1e+8)
  #plt.plot(losses) # TODO: fix this, something to do with agg (GUI)
  plt.savefig("plots/multitask_plot.png")
  plt.show()

  return model

if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"
  model_path = "models/combo_model.pth"

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

  print("Data Files:")
  print(video_files)
  print(log_files)
  print(annot_files)
  assert len(video_files) == len(log_files)
  assert len(video_files) == len(annot_files)
  model = ComboModel().to(device)

  for i in (t := trange(len(video_files))): # TODO: remove the '-2' after debugging!!!!!
    #t.set_description("Loading from files: %s, %s, %s" % ((base_dir+video_files[i], base_dir+log_files[i], base_dir+annot_files[i])))
    print("Loading from files: %s, %s, %s" % ((base_dir+video_files[i], base_dir+log_files[i], base_dir+annot_files[i])))
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
  model = train(all_frames, all_labels, all_annotations, model)

  # TODO: save model
  save_model(model_path, model)

