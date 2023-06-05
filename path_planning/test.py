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

# TODO: test out the model's output compared to groundtruth (show all trajectories and highlight the most probable)

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

  dataset = PathPlannerDataset("../data/sim/22/")
  loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
  loss = MDNLoss()

  fig = go.FigureWidget()
  fig.add_scatter()
  fig.update_layout(xaxis_range=[-50,50])
  fig.update_layout(yaxis_range=[0,50])

  for i in range(10):
    idx = random.randint(0, len(dataset))
    samp = dataset[idx]
    img, path = samp["image"], samp["path"]

    model = PathPlanner().to(device)
    with torch.no_grad():
      model.eval()
      X = torch.tensor([img,img]).float().to(device)
      out = model(X)
      print(out.shape)
      trajectories, modes = loss._get_trajectory_and_modes(out)

      # TODO: display all paths together + groundtruth (highlight highest probability path)
      for path in trajectories:
        fig.data[0].x = path[:, 0]
        fig.data[0].y = path[:, 1]
        figshow(fig)

      cv2.imshow("DISPLAY", disp_img)
      cv2.waitKey(0)

      dataset.cap.release()
      cv2.destroyAllWindows()
