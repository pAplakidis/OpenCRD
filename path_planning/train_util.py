#!/usr/bin/env python3
import os, sys
import cv2
import numpy as np
from tqdm import tqdm
from datetime import date

import io
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils

from util import *
from model import *
from renderer import *

# net input resolution
W = 224
H = 224

# display resolution
d_W = 1920 // 2
d_H = 1080 // 2


class PathPlannerDataset(Dataset):
    def __init__(self, base_dir):
      super(Dataset, self).__init__()
      self.base_dir = base_dir  # TODO: for now just use one clip
      self.video_path = base_dir + "video.mp4"
      self.poses_path = base_dir + "poses.npy"
      self.framepath_path = base_dir + "frame_paths.npy"

      # load meta-data (poses, paths, etc)
      self.poses = np.load(self.poses_path)
      self.frame_paths = np.load(self.framepath_path)
      self.local_poses, self.local_path, self.local_orientations = get_relative_poses(self.poses)
      #print(self.local_path.shape)
      print("Frame Paths (2D):", self.frame_paths.shape)

      # load video
      self.cap = cv2.VideoCapture(self.video_path)

    def __len__(self):
      return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - LOOKAHEAD

    def __getitem__(self, idx):
      #self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx-1)
      self.cap.set(1, idx)
      ret, frame = self.cap.read()
      frame = cv2.resize(frame, (W,H))
      frame = np.moveaxis(frame, -1, 0)
      #return {"image": frame, "path": self.local_path[idx:LOOKAHEAD+idx]}  # TODO: use path for now, later on predict poses
      return {"image": frame, "path": self.frame_paths[idx]}


class Trainer:
  def __init__(self, device, model, train_loader, val_loader, model_path, writer_path=None, early_stop=False):
    self.early_stop = early_stop
    self.model_path = model_path
    if not writer_path:
      today = str(date.today())
      writer_path = "runs/" + today
    print("[+] Tensorboard output path:", writer_path)

    self.writer = SummaryWriter(writer_path)
    self.device = device
    print("[+] Device:", self.device)
    self.model = model.to(self.device)
    self.train_loader = train_loader
    self.val_loader = val_loader

  def save_checkpoint(state, path):
    torch.save(state, path)
    print("Checkpoint saved at", path)

  def train(self, epochs=100, lr=1e-3, path=None):
    #loss_func = nn.MSELoss()
    loss_func = MTPLoss(self.model.n_paths)
    optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    def eval(val_losses, train=False):
      print("[+] Evaluating ...")
      with torch.no_grad():
        try:
          self.model.eval()
          l_idx = 0
          for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
            X = sample_batched['image'].float().to(self.device)
            Y = sample_batched['path'].float().to(self.device)

            out = self.model(X)
            loss = loss_func(out, Y)

            if not train:
              self.writer.add_scalar('evaluation loss', loss.item(), l_idx)
            val_losses.append(loss.item())
            t.set_description("Batch Loss: %.2f"%(loss.item()))
            l_idx += 1

        except KeyboardInterrupt:
          print("[~] Evaluation stopped by user")
      print("[+] Evaluation Done")
      return val_losses

    losses = []
    vlosses = []

    try:
      print("[+] Training ...")
      l_idx = 0
      for epoch in range(epochs):
        self.model.train()
        print("[->] Epoch %d/%d"%(epoch+1, epochs))
        epoch_losses = []
        epoch_vlosses = []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          X = sample_batched['image'].float().to(self.device)
          Y = sample_batched['path'].float().to(self.device)

          optim.zero_grad()
          out = self.model(X)
          #print("Model output: ", out.shape)
          #print("Ground Truth: ", Y.shape)
          loss = loss_func(out, Y)

          self.writer.add_scalar("running loss", loss.item(), l_idx)
          epoch_losses.append(loss.item())

          loss.backward()
          optim.step()

          t.set_description("Batch Training Loss: %.2f"%(loss.item()))
          l_idx += 1

        avg_epoch_loss = np.array(epoch_losses).mean()
        losses.append(avg_epoch_loss)
        print("[=>] Epoch average training loss: %.4f"%(avg_epoch_loss))

        if self.early_stop:
          epoch_vlosses = eval(epoch_vlosses, train=True)
          avg_epoch_vloss = np.array(epoch_vlosses).mean()
          vlosses.append(avg_epoch_vloss)

    except KeyboardInterrupt:
      print("[~] Training stopped by user")
    print("[+] Training Done")
    save_model(self.model_path, self.model)

    for idx, l in enumerate(losses):
      self.writer.add_scalar("final training loss", l, idx)


if __name__ == "__main__":
  #renderer = Renderer3D(RW, RH)
  dataset = PathPlannerDataset("../data/sim/22/")
  print(len(dataset))
  samp = dataset[100]
  img, path = samp["image"], samp["path"]
  print(img.shape)
  print(path.shape)

  # plot path
  fig = go.FigureWidget()
  fig.add_scatter()
  fig.update_layout(xaxis_range=[-50,50])
  fig.update_layout(yaxis_range=[0,50])
  fig.data[0].x = path[:, 0]
  fig.data[0].y = path[:, 1]
  figshow(fig)

  disp_img = np.moveaxis(img, 0, -1)
  disp_img = cv2.resize(disp_img, (d_W,d_H))
  print(disp_img.shape)

  #renderer.draw(path)

  #draw_path(path, disp_img)
  cv2.imshow("DISPLAY", disp_img)
  cv2.waitKey(0)

  dataset.cap.release()
  cv2.destroyAllWindows()
