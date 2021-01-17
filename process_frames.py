#!/usr/bin/env python3
import numpy as np
import cv2

W = 1920//2
H = 1080//2

def get_video_data(path):
  frames = []

  cap = cv2.VideoCapture(path)
  count = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      frame = cv2.resize(frame, (W,H))
      print("Frame", count)
      print(frame)
      print(frame.shape)
      if count % 30 == 0:
        frames.append(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      count += 1
    else:
      break
  cap.release()
  cv2.destroyAllWindows()

  return np.array(frames)


if __name__ == '__main__':
  # TODO: open multiple videos and store frames (skip some since they are almost identical)
  data_path = "data/videos/with_crossroads/"
  filename = "production ID_4370833.mp4"

  # TODO: save frames and label them
  frames = get_video_data(data_path + filename)
  print(frames.shape)

