#!/usr/bin/env python3
import sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt

def extract_polylines(filename):
  tree = ET.parse(filename) # read in the XML
  polylines = []
  for polyline in tree.iter(tag='polyline'):
    frame = polyline.get("frame")
    points = polyline.get("points").split(";")
    for i in range(len(points)):
      points[i] = points[i].split(",")
      for j in range(len(points[i])):
        points[i][j] = float(points[i][j])
    data = (int(frame), points)
    polylines.append(data)

  return sorted(polylines)

# returns frames with annotations
def extract_frame_lines(polylines):
  n_frames = polylines[-1][0]
  frames = []

  # TODO: O(n^2), refactor this
  for i in range(n_frames+1):
    frame = []
    for polyline in polylines:
      if polyline[0] == i:
        frame.append(polyline[1])
    frames.append(sorted(frame))
  
  return frames

# TODO: need to project polylines from coefficients as well (maybe use matplotlib on top of cv2 image) OR create new points from coefficients (since network will output only coefficients)
def extract_coefficients(annotations):
  coefficients = []
  for frame in annotations:
    frame_coef = []
    for polyline in frame:
      polyline = np.array(polyline)
      x, y = polyline.T[0], polyline.T[1]
      coef = np.polyfit(x, y, 2)  # extract coefficients from 2nd degree polyilne points
      frame_coef.append(coef)
    coefficients.append(frame_coef)
  return np.array(coefficients, dtype=object)

# returns a, b, c coefficients of specific polyline in specific frame
def get_coefficients(coefficients, frame, line):
  return coefficients[frame][line]

# helps plot lines using coefficients
def poly_coefficients(x, coeffs):
  y = 0
  for i in range(len(coeffs)):
    y += coeffs[i]*x**i
  return y

# TODO: to make this work with coefs instead of points we need to have a start and a finish for every line (like in the PolyLaneNet paper)
def plot_coefs(frame):
  for coeffs in frame:
    x = np.linspace(0, 9, 10)
    plt.plot(x, poly_coefficients(x, coeffs))
  plt.show()

def display(video_file, annotations):
  cap = cv2.VideoCapture(video_file)
  idx = 0
  while True:
    ret, frame = cap.read()
    if ret:
      print("[+] Frame:", idx)
      polylines = annotations[idx]
      for polyline in polylines:
        polyline = np.array(polyline) # get points for every polyline
        print("Polyline:")
        print(polyline)
        x, y = polyline.T[0], polyline.T[1]
        coefficients = np.polyfit(x, y, 2)  # extract coefficients from 2nd degree polyilne points
        print("Coefficients:")
        print(coefficients)
        frame = cv2.polylines(frame, np.int32([polyline]), False, (0, 0, 255))
      # TODO: need to find a way to resize the polylines to a low resolution to multitask-train with the 320x160-input CrossroadDetection model
      frame = cv2.resize(frame, (1920//2, 1080//2))
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xff == ord('q'):
        break
      idx += 1
    else:
      break
  cap.release()
  cv2.destroyAllWindows()

def draw_polylines(frame, polylines, color=(0, 0, 255)):
  for polyline in polylines:
    polyline = np.array(polyline)
    x, y = polyline.T[0], polyline.T[1]
    frame = cv2.polylines(frame, np.int32([polyline]), False, color, 2)
  return frame

# converts current annotations to new resolution
def convert_annotations(old_res, new_res, annotations):
  W, H = old_res
  new_W, new_H = new_res
  new_annotations = []
  for polylines in annotations:
    new_polylines = []
    for polyline in polylines:
      new_polyline = []
      for point in polyline:
        x, y = point
        new_x = (x*new_W) / W
        new_y = (y*new_H) / H
        new_polyline.append((new_x,new_y))
      new_polylines.append(new_polyline)
    new_annotations.append(new_polylines)
  return np.array(new_annotations, dtype=object)

# converts predicted polylines to new resolution
# TODO: this might be buggy (check the shape of polylines on old and new resolutions)
def convert_polylines(old_res, new_res, polylines):
  W, H = old_res
  new_W, new_H = new_res
  new_polylines = []
  for polyline in polylines:
    new_polyline = []
    for point in polyline:
      x, y = point
      new_x = (x*new_W) / W
      new_y = (y*new_H) / H
      new_polyline.append((new_x,new_y))
    new_polylines.append(new_polyline)
  return np.array(new_polylines)

# TODO: this algorithm has bad complexity (O(n^3)), refactor if possible
# convert polylines per frame to net output vector (flattens the array)
def serialize_polylines(polylines, n_coords, n_points, max_n_lines):
  # check if we have more than n_points
  # TODO: instead of removing the whole line, just get polyline[:n_points]
  for polyline in polylines:
    if len(polyline) != n_points:
      polylines.remove(polyline)
  assert len(polylines) <= max_n_lines, "More than max number of lines found"

  # fill the gaps with negative values (-1 == NULL => out of bounds)
  if len(polylines) < max_n_lines:
    for i in range(max_n_lines - len(polylines)):
      new_polyline = []
      for j in range(n_points):
        point = []
        for k in range(n_coords):
          point.append(-1.0)
        new_polyline.append(point)
      polylines.append(new_polyline)
      
  # flatten
  ret = []
  for i in range(max_n_lines):
    for j in range(n_points):
      for k in range(n_coords):
        ret.append(polylines[i][j][k])

  return np.array(ret)

# TODO: this needs more work depending on the net output, since it is tested only on annotations
# convert network output vector to polylines per frame
def deserialize_polylines(net_output, n_coords, n_points, max_n_lines):
  polylines = []
  point = []
  line = []
  for i in range(len(net_output)):
    point.append(net_output[i])
    if len(point) == 2:
      line.append(point)
      point = []
    if len(line) == 4:
      polylines.append(line)
      line = []

  # remove (-1, -1)/out-of-bounds points from lines
  for polyline in polylines:
    while [-1., -1.] in polyline:
      polyline.remove([-1., -1.]) # TODO: remove all negative numbers, not just (-1., -1.) pairs

  # remove empty lists
  while [] in polylines:
    polylines.remove([])

  return np.array(polylines)

# TODO: polylines' number of points need to be the same (for example 4, 6 might be better)
if __name__ == '__main__':
  video_file = sys.argv[1]
  annotations_file = sys.argv[2]
  polylines = extract_polylines(annotations_file)
  annotations = extract_frame_lines(polylines)
  #display(video_file, annotations)
  coefficients = extract_coefficients(annotations)
  #print(get_coefficients(coefficients, 900, 1))
  #plot_coefs(coefficients[900])

