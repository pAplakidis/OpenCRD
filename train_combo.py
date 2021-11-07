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
from polylines import *


# DATA FUNCTIONS
# get training data from path
def get_data(video_path, log_path, annotations_path, pathp_path, desires_path):
  # get video frames
  frames = pims.Video(video_path, format="mp4")
  
  with open(log_path, 'r') as log_file:
    labels = log_file.read().split("\n")[:-1]
    log_file.close()
  
  # get road edges data
  annotations = extract_polylines(annotations_path)
  annotations = extract_frame_lines(annotations)
  annotations = convert_annotations((annot_W,annot_H), (W,H), annotations)

  # get path data
  path = extract_polylines(pathp_path)
  path = extract_frame_lines(path)
  path = convert_annotations((path_W,path_H), (W,H), path)
  
  desires = get_desires(desires_path)

  return frames, np.array(labels).astype(np.float), annotations, path, desires

def train(frames, labels, annotations, path, desires, model):
  loss_function = ComboLoss(3, model, device)
  optim = torch.optim.Adam(model.parameters(), lr=0.001)
  
  # TODO: we are dealing with more than 1 tasks, so accuracies might be different
  losses, cr_accuracies, re_accuracies = [], [], []
  epochs = 50
  if model.num_layers == 50:
    BS = 8
  else:
    BS = 32
  
  print("[+] Training model ...")
  for epoch in range(epochs):
    print("[+] Epoch", epoch+1, "/", epochs)
    epoch_losses = []
    for i in (t := trange(0, len(frames)-BS, BS)):
      X_train = []  # frame
      Y1_train = []  # crossroad bool
      Y2_train = []  # road edge points
      Y3_train = []  # path points
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
        flat_path = serialize_polylines(path[j], model.n_coords, model.n_points, model.max_n_lines)
        Y3_train.append(flat_path)

      # TODO: this is a temp hack since all desires are 0
      # CHANGE THIS, put it in the above loop to match the rest of the data
      desire = desires[i:i+BS].tolist()
      desire = one_hot_encode(desire)
      desire = torch.tensor(desire).float().to(device)

      X = torch.tensor(np.array(X_train)).float().to(device)
      Y1 = torch.tensor(np.array(Y1_train)).float().to(device)
      Y2 = torch.tensor(np.array(Y2_train)).float().to(device)
      Y3 = torch.tensor(np.array(Y3_train)).float().to(device)

      # forward and backpropagate
      optim.zero_grad()
      out = model(X, desire)

      #print(out[1].shape)
      #print(Y2.shape)
      loss = loss_function(out, Y1, Y2, Y3)
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
  #plt.ylim(-1e+8, 1e+8)
  plt.plot(losses)
  plt.savefig("plots/multitask_plot.png")
  plt.show()

  output_names = ['crossroad', 'road_edges']
  save_onnx("models/combo_model.onnx", model, X, output_names)

  return model

if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"
  model_path = "models/combo_model.pth"

  video_files = []
  log_files = []
  annot_files = []
  path_files = []
  desire_files = []

  for f in listdir(base_dir):
    if f.endswith(".mp4"):
      video_files.append(f)
    elif f.endswith("annotations.xml"):
      annot_files.append(f)
    elif f.endswith("path.xml"):
      path_files.append(f)
    elif f.endswith("desire.txt"):
      desire_files.append(f)
    elif f.endswith(".txt"):
      log_files.append(f)
  video_files, log_files, annot_files, path_files, desire_files = sorted(video_files), sorted(log_files), sorted(annot_files), sorted(path_files), sorted(desire_files)

  video_files = video_files[:3] # TODO: this is a temp hack, need to get all videos' annotations
  log_files = log_files[:len(video_files)]

  print("Data Files:")
  print(video_files)
  print(log_files)
  print(annot_files)
  print(path_files)
  print(desire_files)
  assert len(video_files) == len(log_files)
  assert len(video_files) == len(annot_files)
  assert len(video_files) == len(path_files)
  assert len(video_files) == len(desire_files)
  model = ComboModel(num_layers=34).to(device).train()  # CHANGE THIS

  #for i in (t := trange(len(video_files))):
  for i in (t := trange(len(video_files))):
    #t.set_description("Loading from files: %s, %s, %s" % ((base_dir+video_files[i], base_dir+log_files[i], base_dir+annot_files[i])))
    print("Loading from files: %s, %s, %s, %s, %s" % ((base_dir+video_files[i], base_dir+log_files[i], base_dir+annot_files[i], base_dir+path_files[i], base_dir+desire_files[i])))
    frames, labels, annotations, path, desires = get_data(base_dir+video_files[i], base_dir+log_files[i], base_dir+annot_files[i], base_dir+path_files[i], base_dir+desire_files[i])
    frames = conv_frames(frames)
    if i == 0:
      all_frames = frames
      all_labels = labels
      all_annotations = annotations
      all_paths = path
      all_desires = desires
    else:
      all_frames = np.concatenate((all_frames, frames), axis=0)
      all_labels = np.concatenate((all_labels, labels), axis=0)
      all_annotations = np.concatenate((all_annotations, annotations), axis=0)
      all_paths= np.concatenate((all_paths, path), axis=0)
      all_desires= np.concatenate((all_desires, desires), axis=0)
      
  frames, labels = [], []  # free up some memory
  model = train(all_frames, all_labels, all_annotations, all_paths, all_desires, model)

  save_model(model_path, model)

