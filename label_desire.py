#!/usr/bin/env python3
import sys
import cv2
import numpy as np

# (3, 960, 1280) inputs images
W = 1280
H = 960

# TODO: implement timestamps (so we can go back and forth wherever we want in the video, this might need extra work with the line indices so they can match the frames)

# for every frame/line output the label in a txt file
DESIRE = {0: "forward",
          1: "right",
          2: "left"}

if __name__ == '__main__':
  # NOTE: only .mp4 videos
  video_path = sys.argv[1]
  label_path = video_path[:-4]

  cap = cv2.VideoCapture(video_path)
  labels = []
  
  idx = 0
  while True:
    ret, frame = cap.read()
    
    if ret:
      frame = cv2.resize(frame, (W,H))
      print("Frame", idx)
      print(frame.shape)
      cv2.imshow('frame', frame)
      idx += 1

      key = cv2.waitKey(0)
      # if key pressed is 'c' then crossroad detected, if key is 'q' stop, of key is other continue (no crossroad)
      if key & 0xff == ord('w'):
        label = 0
        labels.append(label)
      elif key & 0xff == ord('d'):
        label = 1
        labels.append(label)
      elif key & 0xff == ord('a'):
        label = 2
        labels.append(label)
      elif key & 0xff == ord('q'):
        break
      else:
        label = 0
        labels.append(label)
      print(DESIRE[label])
    else:
      break

  print(labels)
  cap.release()
  cv2.destroyAllWindows()

  # save to file
  label_file = open(label_path + "_desire.txt", "w")
  for label in labels:
    label_file.write(str(label) + "\n")
  print("Labels written to", label_path+"_desire.txt")
  label_file.close()

