#!/usr/bin/env python3
from model import *
from train_util import *
from util import *


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  # get data
  dataset = PathPlannerDataset("../data/sim/8/")
  train_split = int(len(dataset)*0.7) # 70% training data
  val_split = int(len(dataset)*0.3)   # 30% validation data
  train_set, val_set = random_split(dataset, [train_split, val_split])
  train_loader = DataLoader(train_set, batch_size=12, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_set, batch_size=12, shuffle=True, num_workers=0)

  dataset.cap.release()
  cv2.destroyAllWindows()
