import cv2
import pims
import numpy as np

# These are helpful functions used in training scripts

# CONSTANTS
# network input resolution
W = 320
H = 160

# road_edges' resolution
annot_W = 480
annot_H = 320

# paths' resolution
path_W = 2562
path_H = 1440

# dictionaries
LABEL_DICT = {0: "no crossroad", 1: "crossroad"}  # NOTE: no need to change this for 2 classes (argmax still gets us the same results)
DESIRE = {0: "forward",
          1: "right",
          2: "left"}

# display resolution for the app
disp_W = 1920//2
disp_H = 1080//2

# make pims video into actual numpy frames
def conv_frames(frames):
  imgs = []
  print("Getting frames into proper arrays")
  for frame in frames:
    imgs.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (W,H)))
  print("Frames converted to numpy arrays")
  return np.array(imgs)

# TODO: add desire features here (desire will be one-hot vector encoded and concatenated to input after convolution)
def get_desires(path):
  with open(path, 'r') as f:
    desires = f.read().split("\n")[:-1]
    f.close()
  for i in range(len(desires)):
    desires[i] = int(desires[i])
  print("Log for desires read")
  return np.array(desires).astype(np.int)

# specifically for desire (0, 1, 2)
def one_hot_encode(arr):
  for i in range(len(arr)):
    idx = arr[i]
    arr[i] = [0, 0, 0]
    arr[i][idx] = 1
  return arr

