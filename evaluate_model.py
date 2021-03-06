import numpy as np
import cv2
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_cr_detection_model import ConvNet

W, H = 640, 480
LABEL_DICT = {0: "no crossroad", 1: "crossroad"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

video_path = "data/videos/with_crossroads/city_1.mp4"
log_path = video_path[:-4] + ".txt"

with open(log_path, "r") as log_file:
  labels = log_file.read().split("\n")[:-1]
  log_file.close()
print("Log file read")

# load model
model_path = "models/cr_conv_detector_statedict.pth"
model = ConvNet()
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

# Evaluate model
cap = cv2.VideoCapture(video_path)
Y_preds = []

idx = 0
while True:
  ret, frame = cap.read()
  
  if ret:
    frame = cv2.resize(frame, (W,H))
    print("Frame", idx)
    print(frame.shape)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(0)
    if key & 0xff == ord('q'):
      break
    else:
      X_test = np.moveaxis(frame, -1, 0)
      Y_pred = round(model(torch.tensor(X_test).unsqueeze(0).to(device), dim=1).numpy())
      print("Predicted label", Y_pred)
      Y_preds.append(Y_pred)
    idx += 1
  else:
    break

cap.release()
cv2.destroyAllWindows()
Y_preds = np.array(Y_preds)


print((labels == Y_preds).mean())

