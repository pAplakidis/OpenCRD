#!/usr/bin/env python3
import os, sys
import cv2
import random
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
      self.base_dir = base_dir
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
      # TODO: this is a tempfix, we need to cleanup data (either check for nan during data-collection or training)
      if np.isnan(self.frame_paths[idx]).any():
        self.frame_paths[idx] = np.zeros_like(self.frame_paths[idx])
      return {"image": frame, "path": self.frame_paths[idx]}


class MultiVideoDataset(Dataset):
  def __init__(self, base_dir):
    super(Dataset, self).__init__()
    self.base_dir = base_dir
    self.video_paths = []
    self.framepath_paths = []
    for dir in sorted(os.listdir(base_dir)):
      prefix = self.base_dir+dir+"/"
      self.video_paths.append(prefix+"video.mp4")
      self.framepath_paths.append(prefix+"frame_paths.npy")

    self.caps = [cv2.VideoCapture(str(video_path)) for video_path in self.video_paths]
    self.images = [[capid, framenum] for capid, cap in enumerate(self.caps) for framenum in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-LOOKAHEAD+2)]
    self.frame_paths = [np.load(fp) for fp in self.framepath_paths]
    """
    # check length of images and paths
    print("images:")
    cap = 0
    frame_cnt = []
    for i in range(len(self.images)):
      if self.images[i][0] != cap:
        cap += 1
        frame_cnt.append(self.images[i-1][1])
    frame_cnt.append(self.images[-1][1])
    print(frame_cnt)
    print("frame paths:")
    for fp in self.frame_paths:
      print(fp.shape)
    """

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    capid, framenum = self.images[idx]
    cap = self.caps[capid]
    cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
    ret, frame = cap.read()

    frame = cv2.resize(frame, (W,H))
    frame = np.moveaxis(frame, -1, 0)
    path = self.frame_paths[capid][framenum]
    if np.isnan(path).any():
      path = np.zeros_like(path)

    #img_tensor = torch.from_numpy(frame).float()
    #path_tensor = torch.from_numpy(path).float()
    #return {"image": img_tensor, "path": path_tensor}
    return {"image": frame, "path": path}


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
          for i_batch, sample_batched in enumerate((t := tqdm(self.val_loader))):
            X = torch.tensor(sample_batched['image']).float().to(self.device)
            Y = torch.tensor(sample_batched['path']).float().to(self.device)

            out = self.model(X)
            loss = loss_func(out, Y)

            if not train:
              self.writer.add_scalar('evaluation loss', loss.item(), i_batch)
            val_losses.append(loss.item())
            t.set_description("Batch Loss: %.2f"%(loss.item()))

        except KeyboardInterrupt:
          print("[~] Evaluation stopped by user")
      print("[+] Evaluation Done")
      return val_losses

    losses = []
    vlosses = []
    try:
      print("[+] Training ...")
      for epoch in range(epochs):
        self.model.train()
        print("[=>] Epoch %d/%d"%(epoch+1, epochs))
        epoch_losses = []
        epoch_vlosses = []

        for i_batch, sample_batched in enumerate((t := tqdm(self.train_loader))):
          X = torch.tensor(sample_batched['image']).float().to(self.device)
          Y = torch.tensor(sample_batched['path']).float().to(self.device)

          optim.zero_grad()
          out = self.model(X)
          #print("Model output: ", out.shape)
          #print("Ground Truth: ", Y.shape)
          loss = loss_func(out, Y)

          self.writer.add_scalar("running loss", loss.item(), i_batch)
          epoch_losses.append(loss.item())

          loss.backward()
          optim.step()

          t.set_description("Batch Training Loss: %.2f"%(loss.item()))

        avg_epoch_loss = np.array(epoch_losses).mean()
        losses.append(avg_epoch_loss)
        print("[->] Epoch average training loss: %.4f"%(avg_epoch_loss))

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

    # TODO: evaluate model + stats
    val_losses = []
    val_losses = eval(val_losses)
    print("Avg Eval Loss: %.4f"%(np.array(val_losses).mean()))
  
    self.writer.close()
    return self.model


if __name__ == "__main__":
  #renderer = Renderer3D(RW, RH)
  """
  dataset = PathPlannerDataset("../data/sim/22/")
  print(len(dataset))
  samp = dataset[100]
  img, path = samp["image"], samp["path"]
  print(img.shape)
  print(path.shape)
  print(path)

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
  """

  dataset = MultiVideoDataset("../data/sim/")
  print("Frames in dataset:", len(dataset))
  idxs = []
  for _ in range(10):
    idxs.append(random.randint(0, len(dataset)))

  for idx in idxs:
    print("[+] Frame:", idx)
    samp = dataset[idx]
    img, path = samp["image"], samp["path"]
    print(img.shape)
    print(path.shape)

    # plot path
    fig = go.FigureWidget()
    fig = go.FigureWidget()
    fig.add_scatter()
    fig.update_layout(xaxis_range=[-50,50])
    fig.update_layout(yaxis_range=[0,50])
    fig.data[0].x = path[:, 0]
    fig.data[0].y = path[:, 1]
    figshow(fig)

    disp_img = np.moveaxis(img, 0, -1)
    disp_img = cv2.resize(disp_img, (d_W, d_H))

    cv2.imshow("DISPLAY", disp_img)
    cv2.waitKey(0)

  for cap in dataset.caps:
    cap.release()
  cv2.destroyAllWindows()
