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

LABEL_DICT = {0: "no crossroad", 1: "crossroad"}  # NOTE: no need to change this for 2 classes (argmax still gets us the same results)

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


