#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import *
from model import *
from polylines import *


# TODO: maybe put this into functions (make it cleaner)
if __name__ == '__main__':
  # check for nvidia GPU
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)

  # run with MODE="combo" ./app.py ... to use combo model
  mode = os.getenv('MODE')
  if mode == None:
    mode = "single-net"
  print("Running in", mode, "mode")
  if mode == "combo":
    combo = True
  else:
    combo = False

  #eval_path = "data/videos/with_crossroads/city_4.mp4"
  eval_path = sys.argv[1]
  log_path = eval_path[:-4] + ".txt"
  annot_path = eval_path[:-4] + "_annotations.xml"
  pplan_path = eval_path[:-4] + "_path.xml"

  try:
    with open(log_path, "r") as log_file:
      eval_labels = log_file.read().split("\n")[:-1]
      log_file.close()
  except FileNotFoundError:
    eval_labels = None

  if combo == 0:
    # load Crossroad detector model (TODO: when we use multitask learning later, we will get all drawable data just from the model's output, for now we just do it separately)
    #cr_model_path = "models/cr_detector.pth" # CHANGE THIS
    cr_model_path = "models/resnet_cr_detector_local.pth" # CHANGE THIS
    #cr_model = CRDetector()
    cr_model = ResCRDetector(18, ResBlock, image_channels=3)
    cr_model = load_model(cr_model_path, cr_model).to(device)
    cr_model.eval()

    # load Road-Edge detector model
    #re_model_path = "models/re_detector.pth" # CHANGE THIS
    re_model_path = "models/re_detector_bayesian_local.pth" # CHANGE THIS
    #re_model = REDetector()
    re_model = ResREDetector(18, ResBlock, image_channels=3)
    re_model = load_model(re_model_path, re_model).to(device)
    re_model.eval()

    # load Path-Planner model
    #path_planner_path = "models/path_planner.pth"
    path_planner_path = "models/path_planner_desire.pth"
    path_planner = PathPlanner()
    path_planner = load_model(path_planner_path, path_planner).to(device)
    path_planner.eval()
  else:
    combo_model_path = "models/combo_model.pth"
    combo_model = ComboModel(num_layers=34) # CHANGE THIS
    combo_model = load_model(combo_model_path, combo_model).to(device)
    combo_model.eval()

  # for rounding up to a threshold instead of 0.5 (works with torch.where)
  x = torch.ones(2, 1).to(device)
  y = torch.zeros(2, 1).to(device)

  # get road_edges ground truth
  try:
    #annot_path = sys.argv[2]
    polylines = extract_polylines(annot_path)
    annotations = extract_frame_lines(polylines)
    annotations = convert_annotations((annot_W,annot_H), (disp_W,disp_H), annotations)  # convert the 480x320 lines to display resolution
    print("[+] Using ground truth for road_edges")
  except FileNotFoundError:
    annotations = None

  # TODO: display path ground truth!!!
  try:
    path = extract_polylines(pplan_path)
    path = extract_frame_lines(path)
    path = convert_annotations((path_W,path_H), (disp_W,disp_H), path)  # convert path polyline's resolution to a displayable one
    print("[+] Using ground truth for path")
  except FileNotFoundError:
    path = None

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

        # TODO: maybe add controls for desire
        desire_arr = [0, 0]
        desire = one_hot_encode(desire_arr.copy())
        desire = torch.tensor(desire).float().to(device)

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

          Y_pred2 = path_planner(X, desire)
          print("[~] Predicted value for path_planning")
          print(Y_pred2[1])
          pred_path = deserialize_polylines(Y_pred2[1].cpu().detach().numpy(), path_planner.n_coords, path_planner.n_points, path_planner.max_n_lines)
          pred_path = convert_polylines((W,H), (disp_W,disp_H), pred_path)

        # multitask network
        else:
          out = combo_model(X, desire)
          Y_pred = out[0]
          Y_pred1 = out[1]
          Y_pred2 = out[2]

          print("[~] Predicted value for cr_detection:", Y_pred[1].item())
          cat = torch.where(Y_pred >= 0.8, x, y)
          #pred = LABEL_DICT[int(torch.round(Y_pred[1]).item())]  # round to threshold 0.5
          pred = LABEL_DICT[int(cat[1].item())]                   # round to custom threshold (e.g. 0.8)
          conf = Y_pred[1].item()

          print("[~] Predicted value for re_detection")
          print(Y_pred2[1])
          road_edges = deserialize_polylines(Y_pred1[1].cpu().detach().numpy(), combo_model.re_n_coords, combo_model.re_n_points, combo_model.re_max_n_lines)
          road_edges = convert_polylines((W,H), (disp_W,disp_H), road_edges)  # convert the 320x160 lines to display resolution

          print("[~] Predicted value for path planning")
          print(Y_pred2[1])
          pred_path = deserialize_polylines(Y_pred2[1].cpu().detach().numpy(), combo_model.pth_n_coords, combo_model.pth_n_points, combo_model.pth_max_n_lines)
          pred_path = convert_polylines((W,H), (disp_W,disp_H), pred_path)  # convert the 320x160 lines to display resolution

        # NOTE: the rest is just display code
        frames[1] = cv2.resize(frames[1], (disp_W,disp_H))

        # display groud-truth road edges (red color)
        if annotations is not None:
          polylines = annotations[idx]
          frames[1] = draw_polylines(frames[1], polylines)

        # display ground-truth path (blue color)
        if path is not None:
          p_polylines = path[idx]
          frames[1] = draw_polylines(frames[1], p_polylines, color=(255, 0, 0))

        # draw predicted road edges (orange color)
        frames[1] = draw_polylines(frames[1], road_edges, color=(0, 128, 255))

        # draw predicted path (green color)
        frames[1] = draw_polylines(frames[1], pred_path, color=(0, 255, 0))

        # display category text
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (25, 25) 
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        text = "LABEL: %s, %d" % (pred, int(conf * 100))
        frames[1] = cv2.putText(frames[1], text, org, font,  
                          fontScale, color, thickness, cv2.LINE_AA)

        # display desire text
        org = (25, 55)
        color = (255, 0, 0)
        thickness = 2
        text = "DESIRE: %s, %d" % (DESIRE[desire_arr[1]], desire_arr[1])
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
