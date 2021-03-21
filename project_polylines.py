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

def draw_road_edges(frame, polylines):
  pass


if __name__ == '__main__':
  polylines = extract_polylines(sys.argv[1])
  frames = extract_frame_lines(polylines)

  idx = 0
  for frame in frames:
    print("Frame", idx)
    for line in frame:
      print(line)
    print()
    idx += 1

