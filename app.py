#!/usr/bin/env python3
from os import listdir
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import load_model
from model import CRDetector

W = 320
H = 160
LABEL_DICT = {0: "no-crossroad", 1: "crossroad"}

# check for nvidia GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#eval_path = "data/videos/with_crossroads/city_4.mp4"
eval_path = sys.argv[1]
log_path = eval_path[:-4] + ".txt"

with open(log_path, "r") as log_file:
  eval_labels = log_file.read().split("\n")[:-1]
  log_file.close()

model_path = "models/cr_detector.pth"
model = load_model(model_path).to(device)
model.eval()

cap = cv2.VideoCapture(eval_path)
idx = 0
frames = []
while True:
  ret, frame = cap.read()

  if ret:
    frames.append(frame)
    if idx % 2 != 0:
      frame1 = cv2.resize(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB), (W,H))
      frame2 = cv2.resize(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB), (W,H))
      print("Frame:", idx)
      print("[+] Ground Truth", eval_labels[idx], "->", LABEL_DICT[int(eval_labels[idx])])
      
      # forward to model
      X_test1 = np.moveaxis(frame1, -1, 0)
      X_test2 = np.moveaxis(frame2, -1, 0)
      X_test = []
      X_test.append(X_test1)
      X_test.append(X_test2)
      X_test = np.array(X_test)
      X = torch.tensor(X_test).float().to(device)
      Y_pred = model(X)
      print("[~] Predicted", Y_pred[1].item())
      pred = LABEL_DICT[int(torch.round(Y_pred[1]).item())]
      conf = Y_pred[1].item()

      font = cv2.FONT_HERSHEY_SIMPLEX 
      org = (50, 50) 
      fontScale = 1
      color = (0, 0, 255)
      thickness = 2
      text = "LABEL: %s, %d" % (pred, int(conf * 100))
      frames[1] = cv2.putText(frames[1], text, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA)

      cv2.imshow("frame", cv2.resize(frames[1], (1920//2, 1080//2)))
      if cv2.waitKey(1) & 0xff == ord('q'):
        break

      frames = []
    idx += 1
  else:
    break
  
cap.release()
cv2.destroyAllWindows()

