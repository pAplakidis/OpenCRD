#!/usr/bin/env python3
import sys
import numpy as np
from tqdm import trange
import cv2
import pims
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from model import ConvNet

# resolution
W = 320
H = 160
LABEL_DICT = {0: "no crossroad", 1: "crossroad"}

def get_data(video_path, log_path):
  with open(log_path, "r") as log_file:
    labels = log_file.read().split("\n")[:-1]
    log_file.close()
  print("Log file read")

  # make a Video Object instead of loading all frames in memory
  # to access frames just do: frame = frames[n]
  print("Extracting frames ...")
  frames = pims.Video(video_path, format="mp4")
  print("Done extracting frames")

  return frames, np.array(labels).astype(np.int)

def train(frames, Y_train):
  model = ConvNet()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if device:
    model.to(device)

  loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification problem
  #optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0) # TODO: check if momentum is correct + check if ADAM is better
  optim = torch.optim.Adam(model.parameters(), lr=0.001)  # TODO: experiment with different learning rates(lr)

  BS = 32
  losses, accuracies = [], []

  for epoch in range(2):
    print("[+] Epoch", epoch)
    running_loss = 0.0
    for i in (t := trange(len(frames))):

      # TODO: add batch of images (if memory allows it)
      # get data into network
      # TODO: maybe reshape to (3, H, W) instead of (W, H, 3)
      X_train = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (W,H))
      X = torch.tensor(X_train).float().to(device)
      Y = torch.tensor(Y_train[i]).to(device)
      model.zero_grad()

      # forward feed and backpropagation
      out = model(X)
      cat = torch.round(out)
      accuracy = (cat == Y).float().mean()
      loss = loss_function(out, Y)
      loss = loss.mean()
      loss.backward()
      optim.step()

      # print stats
      loss, accuracy = loss.item(), accuracy.item()
      losses.append(loss)
      accuracies.append(accuracy)
      t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

  # plot losses and accuracies
  plt.ylim(-0.1, 1.1)
  plot(losses)
  plot(accuracies)

  """
  # OLD TRAINING SCRIPT (bad memory usage, just kept some helpful notes for batch training)
  for i in (t := trange(1000)):
    samp = np.random.randint(0, len(frames), size=(BS))
    for idx in samp:
      X_train.append(cv2.resize(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB), (W,H)))
    Y = torch.tensor(Y_train[samp]).long()#.to(device)
  """
  
  return model
 

# TODO: evaluate the net + compute gradients in torch (need to split data to train and test first)
def evaluate(model):
  pass


if __name__ == '__main__':
  video_path = sys.argv[1]
  log_path = video_path[:-4] + ".txt"

  frames, labels = get_data(video_path, log_path)

  #model = train(frames, labels)
  #evaluate(model)

  # TODO: we also need to load and retrain the model (so instead of model = ConvNet() we load it)
  # NOTE: since we are training using Adam optimizer, we need to save the optimizer state dict as well and reload that along with the model for retraining with new data
  #model_path = "models/cr_detection_conv_model.pt"
  #torch.save(model, model_path)
