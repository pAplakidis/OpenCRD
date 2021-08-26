#!/usr/bin/env python3
from os import listdir
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import *
from project_polylines import *

# neural network input resolutions
W = 320
H = 160
LABEL_DICT = {0: "no-crossroad", 1: "crossroad"}

# resolution of road_edges annotations
annot_W = 480
annot_H = 320

disp_W = 1920//2
disp_H = 1080//2

# TODO: maybe make this an environment variable or an option (to be set in terminal)
combo = True  # CHANGE THIS (whether to use combo model or separate models)

if __name__ == '__main__':
  # check for nvidia GPU
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  #eval_path = "data/videos/with_crossroads/city_4.mp4"
  eval_path = sys.argv[1]
  log_path = eval_path[:-4] + ".txt"
  annot_path = eval_path[:-4] + "_annotations.xml"

  try:
    with open(log_path, "r") as log_file:
      eval_labels = log_file.read().split("\n")[:-1]
      log_file.close()
  except FileNotFoundError:
    eval_labels = None

  if not combo:
    # load Crossroad detector model (TODO: when we use multitask learning later, we will get all drawable data just from the model's output, for now we just do it separately)
    #cr_model_path = "models/cr_detector.pth" # CHANGE THIS
    cr_model_path = "models/resnet_cr_detector_local.pth" # CHANGE THIS
    #cr_model = CRDetector()
    cr_model = ResCRDetector(18, ResBlock, image_channels=3)
    cr_model = load_model(cr_model_path, cr_model).to(device)
    cr_model.eval()

    #re_model_path = "models/re_detector.pth" # CHANGE THIS
    re_model_path = "models/re_detector_bayesian_local.pth" # CHANGE THIS
    #re_model = REDetector()
    re_model = ResREDetector(18, ResBlock, image_channels=3)
    re_model = load_model(re_model_path, re_model).to(device)
    re_model.eval()
  else:
    combo_model_path = "models/combo_model.pth"
    combo_model = ComboModel(num_layers=34) # CHANGE THIS
    combo_model = load_model(combo_model_path, combo_model).to(device)
    combo_model.eval()

  # for rounding up to a threshold instead of 0.5 (works with torch.where)
  x = torch.ones(2, 1).to(device)
  y = torch.zeros(2, 1).to(device)

  # get polylines
  try:
    #annot_path = sys.argv[2]
    polylines = extract_polylines(annot_path)
    annotations = extract_frame_lines(polylines)
    annotations = convert_annotations((annot_W,annot_H), (disp_W,disp_H), annotations)  # convert the 480x320 lines to display resolution
  except FileNotFoundError:
    annotations = None

  cap = cv2.VideoCapture(eval_path)
  idx = 0
  frames = []

  prev_frame_time = 0
  new_frame_time = 0

  while True:
    ret, frame = cap.read()

    if ret:
      frames.append(frame)
      if idx % 2 != 0:
        frame1 = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB), (W,H))
        frame2 = cv2.resize(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB), (W,H))
        print("Frame:", idx)
        if eval_labels:
          print("[+] CR_Detection Ground Truth:", eval_labels[idx], "->", LABEL_DICT[int(eval_labels[idx])])
        
        # NOTE: this part handles the network's outputs
        # forward to model(s)
        X_test1 = np.moveaxis(frame1, -1, 0)
        X_test2 = np.moveaxis(frame2, -1, 0)
        X_test = []
        X_test.append(X_test1)
        X_test.append(X_test2)
        X_test = np.array(X_test)
        X = torch.tensor(X_test).float().to(device)

        # individual network for each task
        if not combo:
          Y_pred = cr_model(X)
          print("[~] Predicted value for cr_detection:", Y_pred[1].item())
          cat = torch.where(Y_pred >= 0.8, x, y)
          #pred = LABEL_DICT[int(torch.round(Y_pred[1]).item())]  # round to threshold 0.5
          pred = LABEL_DICT[int(cat[1].item())]                   # round to custom threshold (e.g. 0.8)
          conf = Y_pred[1].item()

          Y_pred1 = re_model(X)
          print("[~] Predicted value for re_detection")
          print(Y_pred1[1])
          road_edges = deserialize_polylines(Y_pred1[1].cpu().detach().numpy(), re_model.n_coords, re_model.n_points, re_model.max_n_lines)
          road_edges = convert_polylines((W,H), (disp_W,disp_H), road_edges)  # convert the 320x160 lines to display resolution
        # multitask network
        else:
          out = combo_model(X)
          Y_pred = out[0]
          Y_pred1 = out[1]

          print("[~] Predicted value for cr_detection:", Y_pred[1].item())
          cat = torch.where(Y_pred >= 0.8, x, y)
          #pred = LABEL_DICT[int(torch.round(Y_pred[1]).item())]  # round to threshold 0.5
          pred = LABEL_DICT[int(cat[1].item())]                   # round to custom threshold (e.g. 0.8)
          conf = Y_pred[1].item()

          print("[~] Predicted value for re_detection")
          print(Y_pred1[1])
          road_edges = deserialize_polylines(Y_pred1[1].cpu().detach().numpy(), combo_model.n_coords, combo_model.n_points, combo_model.max_n_lines)
          road_edges = convert_polylines((W,H), (disp_W,disp_H), road_edges)  # convert the 320x160 lines to display resolution

        # NOTE: the rest is just display code
        frames[1] = cv2.resize(frames[1], (disp_W,disp_H))

        # display groud-truth road edges
        if annotations is not None:
          polylines = annotations[idx]
          frames[1] = draw_polylines(frames[1], polylines)

        # draw predicted road edges
        frames[1] = draw_polylines(frames[1], road_edges, color=(0, 128, 255))

        # display category text
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (25, 25) 
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        text = "LABEL: %s, %d" % (pred, int(conf * 100))
        frames[1] = cv2.putText(frames[1], text, org, font,  
                          fontScale, color, thickness, cv2.LINE_AA)

        # display fps
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        print("FPS:", fps)
        frames[1] = cv2.putText(frames[1],"FPS:"+fps, (disp_W - 150, 25), font,
                          fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

        cv2.imshow("frame", frames[1])
        if cv2.waitKey(1) & 0xff == ord('q'):
          break

        frames = []
      idx += 1
    else:
      break
    
  cap.release()
  cv2.destroyAllWindows()

