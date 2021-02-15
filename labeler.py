#!/usr/bin/env python3
import sys
import cv2
import numpy as np

# (3, 960, 1280) inputs images
W = 1280
H = 960

# for every frame/line output the label in a txt file
# 1 = crossroad, 0 = no_crossroad
if __name__ == '__main__':
  # NOTE: only .mp4 videos
  video_path = sys.argv[1]
  data_path = "/".join(video_path.split("/")[:-1]) + "/"
  label_path = video_path[:-4]

  #cap = cv2.VideoCapture(data_path+filename)
  cap = cv2.VideoCapture(video_path)
  labels = []
  
  idx = 0
  while True:
    ret, frame = cap.read()
    
    if ret:
      frame = cv2.resize(frame, (W,H))
      print("Frame", idx)
      cv2.imshow('frame', frame)
      idx += 1

      if "without_crossroads" in data_path:
        label = 0
        labels.append(label)
        print("Label", label)
        if cv2.waitKey(1) & 0xff == ord('q'):
          break
      else:
        key = cv2.waitKey(0)
        # if key pressed is 'c' then crossroad detected, if key is 'q' stop, of key is other continue (no crossroad)
        if key & 0xff == ord('c'):
          label = 1
          labels.append(label)
        elif key & 0xff == ord('q'):
          break
        else:
          label = 0
          labels.append(label)
        print("Label", label)
    else:
      break

  print(labels)
  cap.release()
  cv2.destroyAllWindows()

  # save to file
  label_file = open(label_path + ".txt", "w")
  for label in labels:
    label_file.write(str(label) + "\n")
  print("Labels writen to", label_path+".txt")
  label_file.close()

