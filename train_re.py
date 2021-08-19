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
from project_polylines import *

# DATA FUNCTIONS
# get training data from path
def get_data(video_path, annotations_path):
  # get video frames
  frames = pims.Video(video_path, format="mp4")
  
  # get road edges data
  annotations = extract_polylines(annotations_path)
  annotations = extract_frame_lines(annotations)
  annotations = convert_annotations((annot_W,annot_H), (W,H), annotations)

  return frames, annotations

# make pims video into actual numpy frames
def conv_frames(frames):
  imgs = []
  print("Getting frames into proper arrays")
  for frame in frames:
    imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (W,H)))
  print("Frames converted to numpy arrays")
  return np.array(imgs)

# TRAINING PROCESS
def train(frames, annotations, model):
  #loss_function = nn.L1Loss().to(device) # mean absolute error
  loss_function = nn.MSELoss().to(device) # mean square error
  #loss_function = nn.NLLLoss()  # TODO: NLL for loss (convert to Bayesian Network?), MSE for accuracy

  optim = torch.optim.Adam(model.parameters(), lr=0.01)  # TODO: try new learning rates (maybe increase the rate)
  #optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)

  losses, accuracies = [], []
  # TODO: try different BS and epochs (lr=0.0001, epochs=500)
  # for 0.001 epochs 500 no more than 150 is needed
  epochs = 50
  BS = 64

  for epoch in range(epochs):
    print("[+] Epoch", epoch, "/", epochs)
    t = trange(0, len(frames)-BS, BS)
    epoch_losses = []
    for i in t: 
      # get data into network
      X_train = []
      Y_train = []

      rng = np.random.default_rng()
      samp = rng.choice(len(frames)-1, size=BS, replace=False)
      for j in samp:
        frame = frames[j]
        frame = np.moveaxis(frame, -1, 0) # [batch_size, channels, height, width]
        X_train.append(frame)
        flat_annot = serialize_polylines(annotations[j], model.n_coords, model.n_points, model.max_n_lines)
        Y_train.append(flat_annot)    

      """
      # NOTE: this might work better here, but it is not compatible with multitask learning
      # without random sample
      for j in range(i, i+BS):
        frame = frames[j]
        frame = np.moveaxis(frame, -1, 0)
        X_train.append(frame)
        flat_annot = serialize_polylines(annotations[j], model.n_coords, model.n_points, model.max_n_lines)
        Y_train.append([flat_annot])
      """

      X = torch.tensor(np.array(X_train)).float().to(device)
      Y = torch.tensor(np.array(Y_train)).float().to(device)

      # TODO: handle accuracy
      # forward and back feed
      optim.zero_grad()
      out = model(X)
      #accuracy = (out == Y).float().mean()  # TODO: this might be wrong
      #loss = loss_function(out, Y.long())
      
      #fit_loss = loss_function(out, Y)
      #complexity_loss = model.nn_kl_divergence()
      #loss = fit_loss + complexity_loss
      
      loss = neg_log_likelihood(out, Y)
      loss = loss.mean()
      loss.backward()
      optim.step()

      # print epoch stats
      loss = loss.item()
      #accuracy = accuracy.item()
      #losses.append(loss/100) # /100 so that we can see it in the graph
      epoch_losses.append(loss) # /10000 so that we can see it in the graph
      #accuracies.append(accuracy)
      #t.set_description("loss %.2f accuracy %.2f out %.2f" % (loss, accuracy, out.mean().item()))
      t.set_description("loss %.2f out %.2f" % (loss, out.mean().item()))

    # add epoch's mean loss to losses (we care more about epoch loss)
    losses.append(np.array(epoch_losses).mean())

  # plot losses and accuracies
  #plt.ylim(-1.0, 1.0)
  plt.ylim(-1e+8, 1e+8)
  plot(losses)
  #plot(accuracies)
  #print(out)
  return model

# TODO: evaluate model
def evalutate(mode, X_test, Y_test):
  pass

# save model for later retraining
def save_model(path, model):
  torch.save(model.state_dict(), path)
  print("Model saved to path", path)

# load model
def load_model(path):
  #model = REDetector()
  model = ResREDetector(18, ResBlock, image_channels=3)
  model.load_state_dict(torch.load(path))
  #model.train()
  model.eval()
  print("Loaded model from", path)
  return model


if __name__ == '__main__':
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  base_dir = "data/videos/usable/"
  model_path = "models/re_detector_bayesian_test.pth"

  video_files = []
  annot_files = []
  for f in listdir(base_dir):
    if f.endswith(".mp4"):
      video_files.append(f)
    elif f.endswith(".xml"):
      annot_files.append(f)
  video_files, annot_files = sorted(video_files), sorted(annot_files)

  video_files = video_files[:3] # TODO: this is a temp hack, need to get all videos' annotations
  print(video_files)
  print(annot_files)

  assert len(video_files) == len(annot_files), "Number of video files != number of annotation files"

  """
  # test polyline functions (this will be later used on the training loop for Y_train)
  n_coords = 2  # 2 coordinates: x,y
  n_points = 4  # number of points of each polyline
  max_n_lines = 6 # max number of polylines per frame

  frames, annotations = get_data(base_dir+video_files[2], base_dir+annot_files[2])

  idx = 0
  for polylines in annotations:
    ret = serialize_polylines(polylines, n_coords, n_points, max_n_lines)
    print("++++++++++")
    print("Frame", idx)
    print(ret)
    print("==========")
    new_polylines = deserialize_polylines(ret, n_coords, n_points, max_n_lines)
    print(new_polylines)
    idx += 1
  """

  #model = REDetector().to(device).train()
  model = ResREDetector(18, ResBlock, image_channels=3).to(device).train()

  # TODO: check if we can overfit a small part of the data (if not,model_path = "../models/re_detector_bayesian_local.pth" then need more layers)
  # NOTE: network might be bad cause of data (sometimes polylines[0] is not left road edge, which causes outliers)
  for i in trange(0, len(video_files)):
    print("[~] Loading from files: %s , %s" % (base_dir+video_files[i], base_dir+annot_files[i]))
    frames, annotations = get_data(base_dir+video_files[i], base_dir+annot_files[i])
    frames = conv_frames(frames)  # TODO: convert all input videos to 320*160 so we don't have to waste time downscaling frames
    if i == 0:
      all_frames = frames
      all_annotations = annotations
    else:
      all_frames = np.concatenate((all_frames, frames), axis=0)
      all_annotations = np.concatenate((all_annotations, annotations), axis=0)

  frames, labels = [], [] # free up memory
  print("[+] Training model ...")
  model = train(all_frames, all_annotations[:-1], model)
  print("[+] Trained model on all data files")

  save_model(model_path, model)
  #model = load_model(model_path).to(device)
  #print(model)

  """
  eval_path = "/content/drive/MyDrive/OpenCRD_dataset/city_1.mp4"
  annot_path = eval_path[:-4] + "_annotations.xml"

  frames, annotations = get_data(eval_path, annot_path)
  idx = 500

  n_coords, n_points, max_n_lines = model.n_coords, model.n_points, model.max_n_lines
  frame1 = cv2.resize(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB), (W,H))
  frame2 = cv2.resize(cv2.cvtColor(frames[idx+1], cv2.COLOR_BGR2RGB), (W,H))
  eval_frame1 = cv2.resize(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB), (W,H))
  eval_frame2 = cv2.resize(cv2.cvtColor(frames[idx+1], cv2.COLOR_BGR2RGB), (W,H))
  cv2_imshow(draw_polylines(frame1, annotations[idx]))
  cv2_imshow(draw_polylines(frame2, annotations[idx+1]))
  print('============================================')
  cv2_imshow(draw_polylines(frame1, deserialize_polylines(serialize_polylines(annotations[idx], n_coords, n_points, max_n_lines), n_coords, n_points, max_n_lines)))
  cv2_imshow(draw_polylines(frame2, deserialize_polylines(serialize_polylines(annotations[idx+1], n_coords, n_points, max_n_lines), n_coords, n_points, max_n_lines)))
  print("Frame:", idx)
  print("Polylines:", annotations[idx])
  print("Frame:", idx+1)
  print("Polylines:", annotations[idx+1])

  # forward to model
  X_test1 = np.moveaxis(frame1, -1, 0)
  X_test2 = np.moveaxis(frame2, -1, 0)
  X_test = []
  X_test.append(X_test1)
  X_test.append(X_test2)
  X_test = np.array(X_test)
  X = torch.tensor(X_test).float().to(device)

  model.eval()
  Y_pred = model(X)
  print("Frame", idx, "predicted:")
  print(Y_pred[0])
  polylines1 = deserialize_polylines(Y_pred[0].cpu().detach().numpy(), model.n_coords, model.n_points, model.max_n_lines)
  print(polylines1)
  cv2_imshow(draw_polylines(eval_frame1, polylines1))
  print("Frame", idx+1, "predicted:")
  print(Y_pred[1])
  polylines2 = deserialize_polylines(Y_pred[1].cpu().detach().numpy(), model.n_coords, model.n_points, model.max_n_lines)
  print(polylines2)
  cv2_imshow(draw_polylines(eval_frame2, polylines2))
  """
