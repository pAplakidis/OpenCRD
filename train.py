#!/usr/bin/env python3
from os import listdir
from tqdm import trange
import numpy as np
import cv2
import pims
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CRDetector


#resolution and labels
#W = 320
#H = 160
W = 480
H = 320
LABEL_DICT = {0: "no crossroad", 1: "crossroad"}


# Get data from files
def get_data(video_path, log_path):
  # for txt file (binary classification)
  with open(log_path, "r") as log_file:
    labels = log_file.read().split("\n")[:-1]
    log_file.close()
  print("Log file read")

  # make a Video Object instead of loading all frames in memory
  print("Extracting frames ...")
  frames = pims.Video(video_path, format="mp4")
  print("Frames extracted")

  return frames, np.array(labels).astype(np.float)

# make pims video into actual numpy frames
def conv_frames(frames):
  imgs = []
  print("Getting frames into proper arrays")
  for frame in frames:
    imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (W,H)))
  print("Frames converted to numpy arrays")
  return np.array(imgs)

# train the network
def train(frames, labels, model):
  loss_function = nn.BCELoss()  # for binary classification
  optim = torch.optim.Adam(model.parameters(), lr=0.001) 

  losses, accuracies = [], []
  epochs = 11
  #BS = 128
  BS = 64
  
  # for rounding up to a threshold instead of 0.5 (works with torch.where)
  x = torch.ones(128, 1).to(device)
  y = torch.zeros(128, 1).to(device)

  for epoch in range(epochs):
    print("[+] Epoch", epoch)
    t = trange(0, len(frames) - BS, BS)  # trange progress bar
    visited_idx = np.array([])

    for i in t:
      # get data into network
      rng = np.random.default_rng()
      samp = rng.choice((frames), size=BS, replace=False)
      visited_idx = np.concatenate((visited_idx, samp)) # add the new random values to visited indices

      # TODO: we must not sample the same frames in the same epoch (use visited_idx)
      X_train = []
      Y_train = []
      for j in samp:
        frame = frames[j]
        frame = np.moveaxis(frame, -1, 0) # [batch_size, channels, height, width]
        X_train.append(frame)
        Y_train.append([labels[j]])
      X = torch.tensor(np.array(X_train)).float().to(device)
      Y = torch.tensor(np.array(Y_train)).float().to(device)

      # forward feed and backpropagation
      optim.zero_grad()
      out = model(X)
      #cat = torch.round(out)
      cat = torch.where(out >= 0.8, x, y) # round up to a threshold instead of 0.5
      accuracy = (cat == Y).float().mean()
      loss = loss_function(out, Y)
      loss = loss.mean()
      loss.backward()
      optim.step()

      # print stats
      loss = loss.item()
      accuracy = accuracy.item()
      losses.append(loss/100) # /100 so that we can see it in the graph
      accuracies.append(accuracy)
      t.set_description("loss %.2f accuracy %.2f out %.2f" % (loss, accuracy, out.mean().item()))

  # plot losses and accuracies
  plt.ylim(-0.1, 1.1)
  plot(losses)
  plot(accuracies)

  return model

# evaluate model
def evaluate(model, X_test, Y_test):
  # for rounding up to a threshold instead of 0.5 (works with torch.where)
  x = torch.ones(128, 1).to(device)
  y = torch.zeros(128, 1).to(device)

  BS = 128
  accuracies = []

  for i in trange(0, len(frames) - BS, BS):
    X_test = []
    Y_test = []
    for j in range(BS):
      frame = frames[i+j]
      frame = np.moveaxis(frame, -1, 0) # [batch_size, channels, height, width]
      X_test.append(frame)
      Y_test.append([labels[i+j]])
    X = torch.tensor(np.array(X_test)).float().to(device)
    Y = torch.tensor(np.array(Y_test)).float().to(device)

    out = model.forward(X)
    cat = torch.where(out >= 0.8, x, y) # round up to a threshold instead of 0.5
    accuracy = (cat == Y).float().mean()
    accuracy = accuracy.item()
    accuracies.append(accuracy)
  
  # plot losses and accuracies
  plt.ylim(-0.1, 1.1)
  plot(accuracies)

# save model
def save_model(model, path):
  torch.save(model.state_dict(), model_path)
  print("Model saved to path", model_path)

# load the model
def load_model(path):
  model = CRDetector()
  model.load_state_dict(torch.load(path))
  model.train()  # for training on new dataset
  #model.eval()  # for evaluation/deployment
  return model


if __name__ == '__main__':
  # check for nvidia GPU
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  # get all files to train model on all of them at once
  base_dir = "data/videos/with_crossroads/"
  video_files = []
  log_files = []

  for f in listdir(base_dir):
    if f.endswith(".mp4"):
      video_files.append(f)
    elif f.endswith(".txt"):
      log_files.append(f)
  video_files, log_files = sorted(video_files), sorted(log_files)
  print(video_files)
  print(log_files)

  # train for all files
  # NOTE: if you want to retrain model, comment the CRDetector line and uncomment load_model line
  model = CRDetector().to(device).train()
  #model = load_model(model_path).to(device)

  # TODO: check for oversampling (more 0s than 1s)
  # get all frames and labels and stich them in the same array, then train the network on the whole dataset
  for i in trange(len(video_files)):
    print("[~] Loading from files: %s , %s" % (base_dir+video_files[i], base_dir+log_files[i]))
    frames, labels = get_data(base_dir+video_files[i], base_dir+log_files[i])
    #frames = conv_frames(frames) # NOTE: this is used if the video input is not already (W,H)
    if i == 0:
      all_frames = frames
      all_labels = labels
    else:
      all_frames = np.concatenate((all_frames, frames), axis=0)
      all_labels = np.concatenate((all_labels, labels), axis=0)
  print("[+] Training model ...")
  model = train(all_frames, all_labels, model)
  print("[+] Trained model on all data files")

  model_path = "models/cr_detector.pth"
  save_model(model, model_path)

  # evaluate model
  eval_video_path = "data/videos/with_crossroads/city_1.mp4"  # CHANGE THIS

  model.eval()
  frames, labels = get_data(base_dir+video_files[i], base_dir+log_files[i])
  frames = conv_frames(frames)
  evaluate(model, frames, labels)

