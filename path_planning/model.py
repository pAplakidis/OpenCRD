import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

# MTP code from [https://github.com/daeheepark/PathPredictNusc]

class MDN(nn.Module):
  def __init__(self, in_feats, out_feats, n_gaussians):
    super(MDN, self).__init__()
    self.in_feats = in_feats
    self.out_feats = out_feats
    self.n_gaussians = n_gaussians
    self.pi = nn.Sequential(
      nn.Linear(self.in_feats, self.n_gaussians),
      nn.Softmax(dim=1)
    )
    self.sigma = nn.Linear(self.in_feats, self.out_feats * self.n_gaussians)
    self.mu = nn.Linear(self.in_feats, self.out_feats * self.n_gaussians)

  def forward(self, x):
    pi = self.pi(x)
    sigma = torch.exp(self.sigma(x))
    sigma = sigma.view(-1, self.n_gaussians, self.out_feats)
    mu = self.mu(x)
    mu = mu.view(-1, self.n_gaussians, self.out_feats)
    return pi, sigma, mu


class MTP(nn.Module):
  # n_modes: number of paths output
  # path_len: number of points of each path
  def __init__(self, in_feats, n_modes=3, path_len=200, hidden_feats=4096):
    super(MTP, self).__init__()
    self.n_modes = n_modes
    self.fc1 = nn.Linear(in_feats, hidden_feats)
    self.fc2 = nn.Linear(hidden_feats, int(n_modes * (path_len*2) + n_modes))

  def forward(self, x):
    x = self.fc2(self.fc1(x))

    # normalize the probabilities to sum to 1 for inference
    mode_probs = x[:, -self.n_modes:].clone()
    if not self.training:
      mode_probs = F.softmax(mode_probs, dim=1)
    
    x = x[:, :-self.n_modes]
    return torch.cat((x, mode_probs), 1)


# NOTE: resolutions not divisible by 8, 16 can cause problems
class PathPlanner(nn.Module):
  def __init__(self, n_paths=3):
    super(PathPlanner, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(pretrained=True)
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))
    #del self.vision.classifier
    """
     (classifier): Sequential(                                                                            
      (0): Dropout(p=0.3, inplace=True)                                                                  
      (1): Linear(in_features=1408, out_features=1000, bias=True)
    """
    # TODO: GRU instead of linear layers
    # TODO: more than image for input (+ desire, recurrent state, etc)

    # multimodal (multiple paths with probabilities) output (check out mixture density networks)
    # meaning output is M future paths (for now) <xi,yi> for i in range(2*H)
    # along with each path's probabilities, these probabilities are passed through a softmax layer
    self.policy = MTP(1408, n_modes=self.n_paths)

  def forward(self, x, desire):
    x = self.vision(x)
    x = x.view(-1, self.num_flat_features(x))
    x = torch.cat((x, desire), 1)
    #print(x.shape)
    x = self.policy(x)
    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


class ComboModel(nn.Module):
  def __init__(self, n_paths=3):
    super(ComboModel, self).__init__()
    self.n_paths = n_paths
    effnet = efficientnet_b2(pretrained=True)

    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))
    self.policy = MTP(1411, n_modes=self.n_paths)
    self.cr_detector = nn.Sequential(
      nn.Linear(1411, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Linear(1024, 128),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Linear(128, 84),
      nn.BatchNorm1d(84),
      nn.ReLU(),
      nn.Linear(84, 1)
    )

  def forward(self, x, desire):
    x = self.vision(x)
    x = x.view(-1, self.num_flat_features(x))
    x = torch.cat((x, desire), 1)
    #print(x.shape)
    path = self.policy(x)
    crossroad = torch.sigmoid(self.cr_detector(x))
    return path, crossroad

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features


# Input: 2 consecutive frames
# Output: trajectory and crossroad prediction
class SuperComboModel(nn.Module):
  def __init__(self, input_size=6, hidden_size=128, n_layers=1, n_paths=3):
    super(SuperComboModel, self).__init__()
    self.n_paths = n_paths
    self.input_size = input_size    # input channels (2 bgr frames -> 2*3 channels)
    self.hidden_size = hidden_size  # output size of GRU unit
    self.n_layers = n_layers        # number of layers in GRU unit
    effnet = efficientnet_b2(pretrained=True)
    effnet.features[0][0] = nn.Conv2d(self.input_size, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    # like MuZero: vision->representation(h), state->dynamics(g), policy->prediction(f)
    self.vision = nn.Sequential(*(list(effnet.children())[:-1]))
    # self.vision = efficientnet_b2(pretrained=True)
    # self.state = nn.GRU(self.vision._fc.in_features, self.hidden_size, self.n_layers, batch_first=True)
    self.state = nn.GRU(1411, self.hidden_size, self.n_layers, batch_first=True)
    self.policy = MTP(self.hidden_size, n_modes=self.n_paths)
    self.cr_detector = nn.Sequential(
      nn.Linear(hidden_size, 64),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )

  def forward(self, in_frames, desire):
    x = torch.cat(in_frames, dim=1)
    x = self.vision(x)
    x = x.view(-1, self.num_flat_features(x))
    x = torch.cat((x, desire), dim=1)
    # print(x.shape)
    out_GRU, pre_GRU = self.state(x)
    # x = out_GRU[:, -1, :] # get the output of the last time step
    path = self.policy(out_GRU)
    crossroad = torch.sigmoid(self.cr_detector(out_GRU))
    return path, crossroad

  def num_flat_features(self, x):
    size = x.size()[1:] # all dimensions except the batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    return num_features

#=============================================================================

#--------------------------------
# CUSTOM LOSS FUNCTIONS
#--------------------------------

def gaussian_probability(sigma, mu, target):
  target = target.unsqueeze(1).expand_as(sigma)
  ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / sigma
  return torch.prod(ret, 2)

# CUSTOM LOSSES
def MDNLoss(pi, sigma, mu, target):
  prob = pi * gaussian_probability(sigma, mu, target)
  nll = -torch.log(torch.sum(prob, dim=1))
  return torch.mean(nll)

# MTPLoss (mutliple-trajectory prediction loss), kinda like Mixture of Experts loss
# L2 Loss for each(i) path predicted
# NOTE: for Multimodal Loss Function do not use Mixture of Experts (ME) Loss,
# but a custom Multiple-Trajectory Prediction Loss:
# get the mode/path m that is closest to the groundtruth
class MTPLoss:
  def __init__(self, n_modes, regression_loss_weigh=1., angle_threshold_degrees=5.):
    self.n_modes = n_modes
    self.n_location_coords_predicted = 2  # (x,y) for each timestep
    self.regression_loss_weight = regression_loss_weigh
    self.angle_threshold = angle_threshold_degrees

  # splits the model predictions into mode probabilities and path
  def _get_trajectory_and_modes(self, model_pred):
    mode_probs = model_pred[:, -self.n_modes:].clone()
    desired_shape = (model_pred.shape[0], self.n_modes, -1, self.n_location_coords_predicted)
    trajectories_no_modes = model_pred[:, :-self.n_modes].clone().reshape(desired_shape)
    return trajectories_no_modes, mode_probs

  # computes the angle between the last points of two paths (degrees)
  @staticmethod
  def _angle_between(ref_traj, traj_to_compare):
    EPSILON = 1e-5

    if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
            ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
        raise ValueError('Both tensors should have shapes (-1, 2).')

    if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
        return 180. - EPSILON

    traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

    # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
    if math.isclose(traj_norms_product, 0):
        return 0.

    # We apply the max and min operations below to ensure there is no value
    # returned for cos_angle that is greater than 1 or less than -1.
    # This should never be the case, but the check is in place for cases where
    # we might encounter numerical instability.
    dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
    angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

    if angle >= 180:
        return angle - EPSILON

    return angle

  # compute the average of l2 norms of each row in the tensor
  @staticmethod
  def _compute_ave_l2_norms(tensor):
    #l2_norms = torch.norm(tensor, p=2, dim=2)
    l2_norms = torch.norm(tensor, p=2, dim=1)
    avg_distance = torch.mean(l2_norms)
    return avg_distance.item()

  # compute angle between the target path and predicted paths
  def _compute_angles_from_ground_truth(self, target, trajectories):
    angles_from_ground_truth = []
    for mode, mode_trajectory in enumerate(trajectories):
        # For each mode, we compute the angle between the last point of the predicted trajectory for that
        # mode and the last point of the ground truth trajectory.
        #angle = self._angle_between(target[0], mode_trajectory)
        angle = self._angle_between(target, mode_trajectory)

        angles_from_ground_truth.append((angle, mode))
    return angles_from_ground_truth

  # finds the index of the best mode given the angles from the ground truth
  def _compute_best_mode(self, angles_from_ground_truth, target, trajectories):
    angles_from_ground_truth = sorted(angles_from_ground_truth)
    max_angle_below_thresh_idx = -1
    for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
      if angle <= self.angle_threshold:
        max_angle_below_thresh_idx = angle_idx
      else:
        break

    if max_angle_below_thresh_idx == -1:
      best_mode = random.randint(0, self.n_modes-1)
    else:
      distances_from_ground_truth = []
      for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx+1]:
        norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :])
        distances_from_ground_truth.append((norm, mode))

      distances_from_ground_truth = sorted(distances_from_ground_truth)
      best_mode = distances_from_ground_truth[0][1]

    return best_mode

  # computes the MTP loss on a batch
  #The predictions are of shape [batch_size, n_ouput_neurons of last linear layer]
  #and the targets are of shape [batch_size, 1, n_timesteps, 2]
  def __call__(self, predictions, targets):
    batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
    trajectories, modes = self._get_trajectory_and_modes(predictions)

    for batch_idx in range(predictions.shape[0]):
      angles = self._compute_angles_from_ground_truth(target=targets[batch_idx], trajectories=trajectories[batch_idx])
      best_mode = self._compute_best_mode(angles, target=targets[batch_idx], trajectories=trajectories[batch_idx])
      best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)

      regression_loss = F.smooth_l1_loss(best_mode_trajectory[0], targets[batch_idx])
      mode_probabilities = modes[batch_idx].unsqueeze(0)
      best_mode_target = torch.tensor([best_mode], device=predictions.device)
      classification_loss = F.cross_entropy(mode_probabilities, best_mode_target)
      
      loss = classification_loss + self.regression_loss_weight * regression_loss
      #deg = abs(math.atan(targets[batch_losses][0][-1][1]/targets[batch_idx][0][-1][0])*180/math.pi)
      #deg_wegith = math.exp(deg/20)
      #loss = loss * deg_weight
      batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)
    
    avg_loss = torch.mean(batch_losses)
    return avg_loss

class ComboLoss(nn.Module):
  def __init__(self, task_num, model, device):
    super(ComboLoss, self).__init__()
    self.task_num = task_num  # TODO: maybe make this constant
    self.model = model
    self.device = device
    self.log_vars = nn.Parameter(torch.zeros((task_num)))

  def forward(self, preds, ground_truths):
    path_pred, cr_pred = preds
    path_gt, cr_gt = ground_truths

    path_loss = MTPLoss(self.model.n_paths)
    cr_loss = nn.BCELoss()

    loss0 = path_loss(path_pred, path_gt)
    loss1 = cr_loss(cr_pred, cr_gt)

    # TODO: need better multitask loss (weighted sum maybe)
    precision0 = torch.exp(-self.log_vars[0])
    #loss0 = precision0*loss0 + self.log_vars[0]

    precision1 = torch.exp(-self.log_vars[1])
    #loss1 = precision1*loss1 + self.log_vars[1]

    loss = loss0 + loss1
    #loss = loss.mean()

    return loss.to(self.device)


def save_model(path, model):
 torch.save(model.state_dict(), path)
 print("Model saved at", path)

def load_model(path, model):
  model.load_state_dict(torch.load(path))
  print("Loaded model from", path)
  return model
