#!/usr/bin/env python3
from model import *
from train_util import *
from util import *

# EXAMPLE USAGE: MODEL_PATH="models/path_planner.pth" WRITER_PATH="runs/test_1" ./train.py

model_path = os.getenv("MODEL_PATH")
if model_path == None:
  model_path = "models/path_planner_desire.pth"
print("[+] Model save path:", model_path)

writer_path = os.getenv("WRITER_PATH")
if writer_path == None:
  writer_path = "runs/train_eval_0"
print("[+] Tensorboard Writer path:", writer_path)

BS = 12
EPOCHS = 100
LR = 1e-2

# TODO: the whole training stack is bottlenecked by the fact that we are loading from the hard-drive instead of RAM
# so the GPU is not utilized completely
if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  # get data
  #dataset = PathPlannerDataset("../data/sim/22/")
  dataset = MultiVideoDataset("../data/sim/train/")
  train_split = int(len(dataset)*0.7) # 70% training data
  val_split = int(len(dataset)*0.3)   # 30% validation data
  train_set, val_set = random_split(dataset, [train_split+1, val_split])
  train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_set, batch_size=BS, shuffle=True, num_workers=0)

  # train model
  #model = PathPlanner()
  #model = ComboModel()
  model = SuperComboModel(n_layers=2)
  print(model)
  trainer = Trainer(device, model, train_loader, val_loader, model_path, writer_path)
  trainer.train(epochs=EPOCHS, lr=LR)

  #dataset.cap.release()
  for cap in dataset.caps:
    cap.release()
  cv2.destroyAllWindows()
