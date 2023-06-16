#!/usr/bin/env python3
import random
import io
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

from train import *
from train_util import *
from util import *
from model import *

def figshow(fig):
  buf = io.BytesIO()
  pio.write_image(fig, buf)
  buf.seek(0)
  file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  cv2.imshow("Predicted Path", img)

if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device:", device)

  #dataset = PathPlannerDataset("../data/sim/22/")
  dataset = MultiVideoDataset("../data/sim/")
  print("Dataset frames: ", len(dataset))
  model = PathPlanner().to(device)
  model = load_model(model_path, model)
  model.eval()
  loss_func = MTPLoss(model.n_paths)
  fig = go.FigureWidget()

  idxs = []
  imgs = []
  paths = []
  desires = []
  for i in range(10):
    idx = random.randint(0, len(dataset))
    samp = dataset[idx]
    img, path, desire = samp["image"], samp["path"], samp["desire"]
    idxs.append(idx)
    imgs.append(img)
    paths.append(path)
    desires.append(desire)

  for i in range(len(idxs)):
    img = imgs[i]
    path = paths[i]
    desire = desires[i]
    print("[+] Frame:", idxs[i])

    disp_img = np.moveaxis(img, 0, -1)
    disp_img = cv2.resize(disp_img, (d_W,d_H))

    with torch.no_grad():
      X = torch.tensor([img,img]).float().to(device)
      DES = torch.tensor([desire, desire]).float().to(device)
      out = model(X, DES)
      trajectories, modes = loss_func._get_trajectory_and_modes(out)

      for idx, pred_path in enumerate(trajectories[0]):
        path_x = pred_path.to("cpu").numpy()[:, 0]
        path_y = pred_path.to("cpu").numpy()[:, 1]
        if modes[0][idx] == torch.max(modes[0]):
          marker = {"color": "red"}
          name = "best_path"
        else:
          marker = {"color": "blue"}
          name = "path"+str(idx)
        fig.add_scatter(x=path_x, y=path_y, name=name, marker=marker)

      fig.add_scatter(x=path[:, 0], y=path[:, 1], name="ground_truth", marker={"color": "green"})
      fig.update_layout(xaxis_range=[-50,50])
      fig.update_layout(yaxis_range=[0,50])
      figshow(fig)
      fig.data = []

      desire_idx = np.argmax(desire)
      print("Desire:", desire_idx, "=>", DESIRE[desire_idx])

      cv2.imshow("DISPLAY", disp_img)
      cv2.waitKey(0)

  #dataset.cap.release()
  for cap in dataset.caps:
    cap.release()
  cv2.destroyAllWindows()
