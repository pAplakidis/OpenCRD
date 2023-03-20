import cv2
import math
import numpy as np

IMG_WIDTH = 1164
IMG_HEIGHT = 874
RW = 1920//2
RH = 1080//2
REC_TIME = 60 # recording length in seconds
LOOKAHEAD = 200

FRAME_TIME = 50

FOCAL = 910.0

# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
  [ 0.,  0.,  1.],
  [ 1.,  0.,  0.],
  [ 0.,  1.,  0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T

# aka 'K' aka camera_frame_from_view_frame
cam_intrinsics = np.array([
  [FOCAL,   0.,   IMG_WIDTH/2.],
  [  0.,  FOCAL,  IMG_HEIGHT/2.],
  [  0.,    0.,     1.]])

# aka 'K_inv' aka view_frame_from_camera_frame
eon_intrinsics_inv = np.linalg.inv(cam_intrinsics)

def to_rotation_matrix(fvector):
  c1 = math.sqrt(fvector[0]**2 + fvector[1]**2)
  s1 = fvector[2]
  c2 = fvector[0] / c1 if c1 else 1.0
  s2 = fvector[2] / c1 if c1 else 0.0

  r = [[fvector[0], -s2, -s1*c2],
       [fvector[1], c2, -s1&s2],
       [fvector[2], 0, c1]]
  return np.array(r)

def to_Rt(t, R):
  Rt = np.eye(4)
  Rt[:3, :3] = R
  Rt[:3, 3] = t
  return Rt

# NOTE: since the data comes from UE4, (x,y,z) needs to be converted to (x,z,y)
# poses = [[location(x,y,z), forward_vector(x,y,z)], [..., ...], ...]
def get_local_poses(poses):
  #print(poses.shape)
  path = poses[:, 0]
  fvector = poses[:, 1]

  # TODO: this is a temp hack, used for data collected before swapping y with z
  #path[:, [2, 1]] = path[:, [1, 2]]
  #fvector[:, [2, 1]] = fvector[:, [1, 2]]

  start_pos = path[0]
  start_rot = fvector[0]

  local_poses = []
  local_path = []
  for i in range(len(path)):
    local_pos = path[i] - start_pos
    local_rot = fvector[i] - start_rot
    local_path.append(local_pos)
    local_poses.append(to_Rt(local_pos, local_rot))

  return np.array(local_poses), np.array(local_path)

def img_from_device(device_path):
  in_shape = device_path.shape
  device_path = np.atleast_2d(device_path)
  path_view = np.einsum('jk, ik->ij', view_frame_from_device_frame, device_path)

  #path_view[path_view[:, 2] < 0] = np.nan  # TODO: commenting this out is a temp hack
  img_path = path_view/path_view[:,2:3]
  return img_path.reshape(in_shape)[:,:2]

def normalize(img_pts):
  # normalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_normalized = eon_intrinsics_inv.dot(img_pts.T).T
  img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
  return img_pts_normalized[:,:2].reshape(input_shape)

def denormalize(img_pts):
  # denormalizes image coordinates
  # accepts single pt or array of pts
  img_pts = np.array(img_pts)
  input_shape = img_pts.shape
  img_pts = np.atleast_2d(img_pts)
  img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0],1))))
  img_pts_denormalized = cam_intrinsics.dot(img_pts.T).T
  img_pts_denormalized[img_pts_denormalized[:,0] > IMG_WIDTH] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,0] < 0] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] > IMG_HEIGHT] = np.nan
  img_pts_denormalized[img_pts_denormalized[:,1] < 0] = np.nan
  return img_pts_denormalized[:,:2].reshape(input_shape)

def draw_path(path, img, width=1, height=1.2, fill_color=(128,0,255), line_color=(0,255,0)):
  # TODO: debug 3D to 2D convertion
  img_points_norm = img_from_device(path) # TODO: this outputs NAN
  img_pts = denormalize(img_points_norm)
  valid = np.isfinite(img_pts).all(axis=1)
  img_pts = img_pts[valid].astype(int)

  print(len(img_pts))
  for i in range(1, len(img_pts)):
    #print(img_pts[i])
    cv2.circle(img, img_pts[i], 1, (0, 0, 255), -1)
