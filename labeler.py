#!/usr/bin/env python3
import cv2
import numpy as np

# NOTE: for every frame/line output the label in a txt file
# NOTE: might not need to split into frames (check comma.ai calib_challenge on how people used mp4 videos for convolutional neural net)

# 1 = crossroad, 0 = no_crossroad
# (3, 960, 1280) inputs images
W = 1280
H = 960

if __name__ == '__main__':
  # TODO: make these work with args
  #data_path = "data/videos/with_crossroads/"
  data_path = "data/videos/without_crossroads/"
  #filename = "city_1.mp4"
  filename = "highway_2.mp4"

  label_path = data_path + "highway_2"

  cap = cv2.VideoCapture(data_path+filename)
  label_file = open(label_path + ".txt", "w")
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
        # if key pressed is 'c' then crossroad detected, if key is 'q' stop, of key is other continue (no crossroad)
        if cv2.waitKey(0) & 0xff == ord('c'):
          label = 1
          labels.append(label)
        elif cv2.waitKey(0) & 0xff == ord('q'):
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
  for label in labels:
    label_file.write(str(label) + "\n")
  print("Labels writen to", label_path+".txt")
  label_file.close()

