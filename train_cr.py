#!/usr/bin/env python3
import numpy as np
import cv2
import pims
from tqdm import trange
from os import listdir

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

from helpers import *
from model import *

#resolution and labels
# TESLA resolution
#W = 1280
#H = 960

# COMMA.AI resolution
#W = 1164
#H = 874

# low resolutions (to avoid low memory problems)
#W = 480
#H = 320

# Get data from files
def get_data(video_path, log_path):
  # NOTE: it turns out that .txt files work for 2 class classification as well

  # for txt file (binary classification)
  with open(log_path, "r") as log_file:
    labels = log_file.read().split("\n")[:-1]
    log_file.close()

  print("Log file read")

  # make a Video Object instead of loading all frames in memory
  # to access frames just do: frame = frames[n]
  print("Extracting frames ...")
  frames = pims.Video(video_path, format="mp4")
  print("Done extracting frames")

  return frames, np.array(labels).astype(np.float)

# train the network
def train(frames, labels, model):
  loss_function = nn.BCELoss()  # for binary classification
  optim = torch.optim.Adam(model.parameters(), lr=0.001) 

  losses, accuracies = [], []
  epochs = 11
  BS = 128
  
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
      samp = rng.choice(len(frames), size=BS, replace=False)
      #visited_idx = np.concatenate((visited_idx, samp)) # add the new random values to visited indices

      # TODO: we must not sample the same frames in the same epoch (use visited_idx)
      X_train = []
      Y_train = []
      for j in samp:
        frame = frames[j]
        frame = np.moveaxis(frame, -1, 0) # [batch_size, channels, height, width]
        X_train.append(frame)
        Y_train.append([labels[j]])
      samp = []
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


if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"
  model_path = "models/resnet_cr_detector_test.pth" # CHANGE THIS

  """
  video_path = "data/videos/usable/city_1.mp4"  # CHANGE THIS
  log_path = video_path[:-3] + "txt"
  #log_path = video_path[:-3] + "log"

  frames, labels = get_data(video_path, log_path)
  print(labels)
  """

  # get all files to train model on all of them at once
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

  assert len(video_files) == len(log_files)

  # train for all files
  #model = ConvNet().to(device).train() # Uncomment this if you want to train from the start, else just run the loader block first
  model = ResCRDetector(18, ResBlock, image_channels=3).to(device).train()

  # TODO: check for oversampling (more 0s than 1s)
  # get all frames and labels and stich them in the same array, then train the network on the whole dataset
  for i in trange(len(video_files)):
    print("[~] Loading from files: %s , %s" % (base_dir+video_files[i], base_dir+log_files[i]))
    frames, labels = get_data(base_dir+video_files[i], base_dir+log_files[i])
    frames = conv_frames(frames)
    if i == 0:
      all_frames = frames
      all_labels = labels
    else:
      all_frames = np.concatenate((all_frames, frames), axis=0)
      all_labels = np.concatenate((all_labels, labels), axis=0)

  frames, labels = [], [] # free up memory
  print("[+] Training model ...")
  model = train(all_frames, all_labels, model)
  print("[+] Trained model on all data files")

  # save model for later retraining
  save_model(model.state_dict(), model_path)

  """
  # load the model
  #model = ConvNet()
  model = ResCRDetector(18, ResBlock, image_channels=3)
  model.load_state_dict(torch.load(model_path))
  model.train()  # for training on new dataset
  #model.eval()  # for evaluation
  model.to(device)
  """

  """
  eval_video_path = "../data/videos/usable/city_1.mp4"  # CHANGE THIS

  model.eval()
  frames, labels = get_data(base_dir+video_files[i], base_dir+log_files[i])
  frames = conv_frames(frames)
  print(frames.shape)
  print(labels.shape)
  evaluate(model, frames, labels)


  eval_path = "/content/drive/MyDrive/OpenCRD_dataset/city_4.mp4"
  log_path = eval_path[:-4] + ".txt"

  with open(log_path, "r") as log_file:
    eval_labels = log_file.read().split("\n")[:-1]
    log_file.close()

  frames = pims.Video(eval_path, format="mp4")
  idx = 1000

  frame1 = cv2.resize(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB), (W,H))
  frame2 = cv2.resize(cv2.cvtColor(frames[idx+1], cv2.COLOR_BGR2RGB), (W,H))
  cv2_imshow(frame1)
  cv2_imshow(frame2)
  print("Frame:", idx)
  print("[+]", eval_labels[idx], "->", LABEL_DICT[int(eval_labels[idx])])
  print("Frame:", idx+1)
  print("[+]", eval_labels[idx+1], "->", LABEL_DICT[int(eval_labels[idx+1])])
      
  # forward to model
  # NOTE: since we are using batch normalization, we need more than 1 images
  X_test1 = np.moveaxis(frame1, -1, 0)
  X_test2 = np.moveaxis(frame2, -1, 0)
  X_test = []
  X_test.append(X_test1)
  X_test.append(X_test2)
  X_test = np.array(X_test)
  X = torch.tensor(X_test).float().to(device)

  model.eval()
  Y_pred = model(X)
  print(Y_pred)
  """

