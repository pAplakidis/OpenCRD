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

def get_data(video_path, annotations_path, desires_path):
  frames = pims.Video(video_path, format="mp4")

  path = extract_polylines(annotations_path)
  path = extract_frame_lines(path)
  path = convert_annotations((path_W,path_H), (W,H), path)

  desires = get_desires(desires_path)

  return frames, path, desires

def train(frames, path, desires, model):
  # TODO: try NLLLoss() or neg_log_likelihood (from model.py)
  loss_function = nn.MSELoss
  optim = torch.optim.Adam(model.parameters(), lr=0.001)  # TODO: lower the learning rate and train on more epochs

  # TODO: handle accuracies as well
  losses = []
  epochs = 50
  BS = 128

  for epoch in range(epochs):
    print("[+] Epoch", epoch+1, "/", epochs)
    epoch_losses = []
    for i in (t := trange(0, len(frames)-BS, BS)):
      # get data into NN
      X_train = []
      Y_train = []

      rng = np.random.default_rng()
      samp = rng.choice(len(frames)-1, size=BS, replace=False)
      for j in samp:
        frame = frames[j]
        frame = np.moveaxis(frame, -1, 0) # [batch_size, channels, height, width]
        X_train.append(frame)
        flat_path = serialize_polylines(path[j], model.n_coords, model.n_points, model.max_n_lines)
        Y_train.append(flat_path)

      desire = desires[i:i+BS].tolist() # TODO: this is not right since we are dealing with a random sample (maybe add them in the loop) (it works temporarily since all values are 0/forward)
      desire = one_hot_encode(desire)
      desire = torch.tensor(desire).float().to(device)
      X = torch.tensor(np.array(X_train)).float().to(device)
      Y = torch.tensor(np.array(Y_train)).float().to(device)

      # forward and backpropagation
      optim.zero_grad()
      out = model(X, desire)
      loss = neg_log_likelihood(out, Y)
      #loss = loss_function(out, Y)
      loss = loss.mean()
      loss.backward()
      optim.step()

      # stats
      loss = loss.item()
      epoch_losses.append(loss)
      t.set_description("loss %.2f out %.2f" % (loss, out.mean().item()))

    losses.append(np.array(epoch_losses).mean())

  # plot losses and accuracies
  plot(losses)
  plt.savefig("plots/path_planner_plot.png")
  plt.show()

  return model


if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"
  #model_path = "models/path_planner.pth"
  model_path = "models/path_planner_desire.pth"

  video_files = []
  path_files = []
  desire_files = []
  for f in listdir(base_dir):
    if f.endswith(".mp4"):
      video_files.append(f)
    elif f.endswith("path.xml"):
      path_files.append(f)
    elif f.endswith("_desire.txt"):
      desire_files.append(f)
  video_files, path_files, desire_files = sorted(video_files), sorted(path_files), sorted(desire_files)

  video_files = video_files[:3] # TODO: this is a temp hack, need to get all videos' annotations
  print(video_files)
  print(path_files)

  assert len(video_files) == len(path_files), "Number of video files != number of annotation files"

  model = PathPlanner().to(device).train()

  for i in trange(0, len(video_files)-1): # TODO: remove the -2 when done debugging
    print("[~] Loading from files: %s , %s" % (base_dir+video_files[i], base_dir+path_files[i]))
    frames, path, desires = get_data(base_dir+video_files[i], base_dir+path_files[i], base_dir+desire_files[i])
    frames = conv_frames(frames)
    #print(path.shape)
    print()
    if i == 0:
      all_frames = frames
      all_paths = path
      all_desires = desires
    else:
      all_frames = np.concatenate((all_frames, frames), axis=0)
      all_paths = np.concatenate((all_paths, path), axis=0)
      all_desires = np.concatenate((all_desires, desires), axis=0)

  #frames, path = [], [] # free up memory
  print("[+] Training model ...")
  model = train(all_frames, all_paths[:-1], all_desires, model)
  print("[+] Trained model on all data files")
  save_model(model_path, model)

