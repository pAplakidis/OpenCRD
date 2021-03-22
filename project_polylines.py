#!/usr/bin/env python3
import sys
import numpy as np
import cv2
import pandas as pd
import xml.etree.ElementTree as ET

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
    frames.append(frame)
  
  return frames

# TODO: need to project polylines from coefficients as well (maybe use matplotlib on top of cv2 image)
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
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xff == ord('q'):
        break
      idx += 1
    else:
      break
  cap.release()
  cv2.destroyAllWindows()


# TODO: polylines' number of points need to be the same (for example 4)
if __name__ == '__main__':
  video_file = sys.argv[1]
  annotations_file = sys.argv[2]
  polylines = extract_polylines(annotations_file)
  annotations = extract_frame_lines(polylines)
  #display(video_file, annotations)
  coefficients = extract_coefficients(annotations)
  print(get_coefficients(coefficients, 900, 1))

