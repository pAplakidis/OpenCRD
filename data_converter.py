#!/usr/bin/env python3
import sys
import numpy as np

with open(sys.argv[1], "r") as txt_file:
  labels = txt_file.read().split("\n")[:-1]
  txt_file.close()

labels = np.array(labels).astype(np.float)
print(labels)

new_labels = []
for label in labels:
  new_label = "1.0, 0.0" if label == 0 else "0.0, 1.0"
  new_labels.append(new_label)
#new_labels = np.array(new_labels)
print(new_labels)

with open(sys.argv[1][:-3]+"log", "w") as log_file:
  for label in new_labels:
    log_file.write(str(label) + "\n")
  log_file.close()
  print("Written to", sys.argv[1][:-3]+"log")

