#!/usr/bin/env python3
from model import *
from train_util import *
from util import *

model_path = "models/path_planner.pth"
writer_path = "runs/train_eval_0"

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  # get data
  dataset = PathPlannerDataset("../data/sim/22/")
  train_split = int(len(dataset)*0.7) # 70% training data
  val_split = int(len(dataset)*0.3)   # 30% validation data
  train_set, val_set = random_split(dataset, [train_split+1, val_split])
  train_loader = DataLoader(train_set, batch_size=12, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_set, batch_size=12, shuffle=True, num_workers=0)

  # train model
  model = PathPlanner()
  print(model)
  trainer = Trainer(device, model, train_loader, val_loader, model_path, writer_path)
  trainer.train()

  dataset.cap.release()
  cv2.destroyAllWindows()
