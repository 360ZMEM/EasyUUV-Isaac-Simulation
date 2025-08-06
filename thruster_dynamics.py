from omni.isaac.lab.utils.math import quat_from_euler_xyz
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np 
import torch

def get_thruster_com_and_orientations(device):
  def create_tf_rpy(x,y,z,rr,rp,ry):
    print(rr,rp,ry)
    shift = torch.Tensor([x, y, z])
    r = quat_from_euler_xyz(torch.Tensor([rr]), torch.Tensor([rp]), torch.Tensor([ry]))[0]
    print(rr, rp, ry, r[0], r[1], r[2], r[3])
    return shift, r

  def create_tf_quat(x,y,z,w,vx,vy,vz):
    shift = torch.Tensor([x, y, z])
    r = torch.Tensor([w, vx, vy, vz])
    return shift, r
  
  def create_tf_rpyquat(x, y, z, roll, pitch, yaw):
    roll = torch.as_tensor(roll)
    pitch = torch.as_tensor(pitch)
    yaw = torch.as_tensor(yaw)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    vx = sr * cp * cy - cr * sp * sy
    vy = cr * sp * cy + sr * cp * sy
    vz = cr * cp * sy - sr * sp * cy

    return create_tf_quat(x, y, z, w, vx, vy, vz)

  length = 0.56 # approximate value, not precise
  width = 0.43   
  height = 0.24  

  thruster_info = {
      "front_left_vertical": create_tf_rpyquat( 
          x = width * 0.3,
          y = length * 0.375,
          z = 0.03,
          roll = 0,
          pitch = -1.5708,
          yaw = 0
      ),
      "front_right_vertical": create_tf_rpyquat( 
          x = width * 0.3,
          y = -length * 0.375,
          z = 0.03,
          roll = 0,
          pitch = -1.5708,
          yaw = 0
      ),
      "rear_left_vertical": create_tf_rpyquat( 
          x = -width * 0.3,
          y = length * 0.375,
          z = 0.03,
          roll = 0,
          pitch = -1.5708,
          yaw = 0
      ),
      "rear_right_vertical": create_tf_rpyquat( 
          x = -width * 0.3,
          y = -length * 0.375,
          z = 0.03,
          roll = 0,
          pitch = -1.5708,
          yaw = 0
      ),
      "front_left_horizontal": create_tf_rpyquat(
          x = width * 0.2,
          y = length * 0.2,
          z = -0.02,
          roll = 0,
          pitch = 0,
          yaw = -0.785398 # -45deg
      ),
      "front_right_horizontal": create_tf_rpyquat(
          x = width * 0.2,
          y = -length * 0.2,
          z = -0.02,
          roll = 0,
          pitch = 0,
          yaw = 0.785398 # 45deg
      ),
      "rear_left_horizontal": create_tf_rpyquat(
          x = -width * 0.2,
          y = length * 0.2,
          z = -0.02,
          roll = 0,
          pitch = 0,
          yaw = -2.356194 # -135deg
      ),
      "rear_right_horizontal": create_tf_rpyquat(
          x = -width * 0.2,
          y = -length * 0.2,
          z = -0.02,
          roll = 0,
          pitch = 0,
          yaw = 2.356194 # 135deg
      )
  }
  # vector pointing from com->thruster location (thruster, 3)

  # new THRUSTER ORDERING IS 
  # 0 - front_left_vertical
  # 1 - front_right_vertical
  # 2 - rear_left_vertical
  # 3 - rear_right_vertical
  # 4 - front_left_horizontal
  # 5 - front_right_horizontal
  # 6 - rear_left_horizontal
  # 7 - rear_right_horizontal

  thruster_com_offsets = torch.tensor([
    [thruster_info["front_left_vertical"][0][0], thruster_info["front_left_vertical"][0][1], thruster_info["front_left_vertical"][0][2]],
    [thruster_info["front_right_vertical"][0][0], thruster_info["front_right_vertical"][0][1], thruster_info["front_right_vertical"][0][2]],
    [thruster_info["rear_left_vertical"][0][0], thruster_info["rear_left_vertical"][0][1], thruster_info["rear_left_vertical"][0][2]],
    [thruster_info["rear_right_vertical"][0][0], thruster_info["rear_right_vertical"][0][1], thruster_info["rear_right_vertical"][0][2]],
    [thruster_info["front_left_horizontal"][0][0], thruster_info["front_left_horizontal"][0][1], thruster_info["front_left_horizontal"][0][2]],
    [thruster_info["front_right_horizontal"][0][0], thruster_info["front_right_horizontal"][0][1], thruster_info["front_right_horizontal"][0][2]],
    [thruster_info["rear_left_horizontal"][0][0], thruster_info["rear_left_horizontal"][0][1], thruster_info["rear_left_horizontal"][0][2]],
    [thruster_info["rear_right_horizontal"][0][0], thruster_info["rear_right_horizontal"][0][1], thruster_info["rear_right_horizontal"][0][2]]
  ])

  # quaternions to go from COM frame to thruster frame (thruster, 4)

  thruster_quats = torch.tensor([
    [thruster_info["front_left_vertical"][1][0], thruster_info["front_left_vertical"][1][1], thruster_info["front_left_vertical"][1][2], thruster_info["front_left_vertical"][1][3]],
    [thruster_info["front_right_vertical"][1][0], thruster_info["front_right_vertical"][1][1], thruster_info["front_right_vertical"][1][2], thruster_info["front_right_vertical"][1][3]],
    [thruster_info["rear_left_vertical"][1][0], thruster_info["rear_left_vertical"][1][1], thruster_info["rear_left_vertical"][1][2], thruster_info["rear_left_vertical"][1][3]],
    [thruster_info["rear_right_vertical"][1][0], thruster_info["rear_right_vertical"][1][1], thruster_info["rear_right_vertical"][1][2], thruster_info["rear_right_vertical"][1][3]],
    [thruster_info["front_left_horizontal"][1][0], thruster_info["front_left_horizontal"][1][1], thruster_info["front_left_horizontal"][1][2], thruster_info["front_left_horizontal"][1][3]],
    [thruster_info["front_right_horizontal"][1][0], thruster_info["front_right_horizontal"][1][1], thruster_info["front_right_horizontal"][1][2], thruster_info["front_right_horizontal"][1][3]],
    [thruster_info["rear_left_horizontal"][1][0], thruster_info["rear_left_horizontal"][1][1], thruster_info["rear_left_horizontal"][1][2], thruster_info["rear_left_horizontal"][1][3]],
    [thruster_info["rear_right_horizontal"][1][0], thruster_info["rear_right_horizontal"][1][1], thruster_info["rear_right_horizontal"][1][2], thruster_info["rear_right_horizontal"][1][3]]
  ])

  return thruster_com_offsets, thruster_quats
