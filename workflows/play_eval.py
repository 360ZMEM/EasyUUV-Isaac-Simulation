import pdb
import argparse
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='EasyUUV-Direct-v1', help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--eval_name", type=str, default='eval_sin', help="Name of the eval run to store in wandb")
parser.add_argument("--custom_weights", type=str, default=None, help="Path to custom weights file")

# Eval parameters
parser.add_argument("--com_cob_offset", type=float, default=0.01, help="Distance of center of buoyancy from the center of mass along the X axis")
parser.add_argument("--volume", type=float, default=0.228, help="Volume of the robot for buoyancy force estimates")
parser.add_argument("--action_noise_std", type=float, default=0., help="Standard deviation of action noise distribution")
parser.add_argument("--observation_noise_std", type=float, default=0., help="Standard deviation of observation noise distribution")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import pandas as pd
import wandb
import numpy as np
import math

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from omni.isaac.lab.utils.math import quat_from_angle_axis, quat_error_magnitude, euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_conjugate, quat_mul

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asymmetric_noise_cfg import *

import datetime
strftime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def angle_remap(angle):
    return (angle + torch.pi) % (2*torch.pi) - torch.pi

def ang_to_quat(roll:np.ndarray, pitch:np.ndarray, yaw:np.ndarray) -> np.ndarray:
    q = np.zeros((len(roll), 4))
    q[:, 0] = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q[:, 1] = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    q[:, 2] = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    q[:, 3] = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    return q

def quat_diff(quat1:np.ndarray, quat2:np.ndarray) -> np.ndarray:
    q1_conjugate = quat1.copy()
    q1_conjugate[:, 1:] *= -1  
    diff = np.zeros_like(quat1)
    for i in range(len(quat1)):
        diff[i, 0] = quat2[i, 0] * q1_conjugate[i, 0] - quat2[i, 1] * q1_conjugate[i, 1] - quat2[i, 2] * q1_conjugate[i, 2] - quat2[i, 3] * q1_conjugate[i, 3]
        diff[i, 1] = quat2[i, 0] * q1_conjugate[i, 1] + quat2[i, 1] * q1_conjugate[i, 0] + quat2[i, 2] * q1_conjugate[i, 3] - quat2[i, 3] * q1_conjugate[i, 2]
        diff[i, 2] = quat2[i, 0] * q1_conjugate[i, 2] - quat2[i, 1] * q1_conjugate[i, 3] + quat2[i, 2] * q1_conjugate[i, 0] + quat2[i, 3] * q1_conjugate[i, 1]
        diff[i, 3] = quat2[i, 0] * q1_conjugate[i, 3] + quat2[i, 1] * q1_conjugate[i, 2] - quat2[i, 2] * q1_conjugate[i, 1] + quat2[i, 3] * q1_conjugate[i, 0]
    return diff
def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )

    env_cfg.domain_randomization.use_custom_randomization = False
    env_cfg.volume = 0.02166# 0.2166 0.228 0.2394
    env_cfg.com_to_cob_offset[0] += 0.02; env_cfg.com_to_cob_offset[0] += 0.02; env_cfg.com_to_cob_offset[0] += 0.025
    # env_cfg.com_to_cob_offset[0] += args_cli.com_cob_offset
    # env_cfg.volume = args_cli.volume

    env_cfg.use_boundaries = False
    env_cfg.cap_episode_length = False
    env_cfg.episode_length_before_reset = 0

    env_cfg.goal_spawn_radius = 0

    env_cfg.eval_mode = True

    env_cfg.control_method = 'Ssurface'
    env_cfg.s_ratio = 4.9
    env_cfg.self_adapt = True
    # PID_PWM_value = 0.6
    # env_cfg.PID_init_args = torch.tensor([[0.6 / PID_PWM_value, 0.08/ PID_PWM_value, 0 / PID_PWM_value], # x-axis rotation, Roll
    #                  [0.6 / PID_PWM_value, 0.08/ PID_PWM_value, 0 / PID_PWM_value], # y-axis rotation, Pitch
    #                  [1.0 / PID_PWM_value, 0.13/ PID_PWM_value, 0 / PID_PWM_value], # z-axis rotation, Yaw
    #                  [0.16 / PID_PWM_value, 0.07/ PID_PWM_value, 0 / PID_PWM_value]],device='cuda:0')

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if not args_cli.custom_weights:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.custom_weights
    
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    wandb.init(
        project=agent_cfg.to_dict()['wandb_project'],
        name=args_cli.eval_name + '_' + strftime,
        config=env_cfg
    )

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load('/home/zmem063/isaaclab/logs/rsl_rl/warpauv_direct/SS4_DRB/model_500.pt') # '/home/zmem063/isaaclab/logs/rsl_rl/warpauv_direct/2025-07-06_22-24-21/model_450.pt'
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create dir to save logs into
    save_path = os.path.join("source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint[:-3] + "_play")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"[INFO]: Saving results into: {save_path}")

    # path for saving csv logs
    eval_csv_path = os.path.join(save_path, "logs.csv")

    # create dataframe to save results into
    log_df = pd.DataFrame(columns=[
        'des_depth', 
        'des_roll', 
        'des_pitch', 
        'des_yaw',
        'true_x',
        'true_y',
        'true_z',
        'true_x_vel',
        'true_y_vel',
        'true_z_vel',
        'true_roll',
        'true_pitch',
        'true_yaw',
        'true_roll_vel',
        'true_pitch_vel',
        'true_yaw_vel',
        'mse',
        'reward'
        ])

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    goal_list = [
        # ([0, 0, 0], [1, 0, 0]),
        # ([0, 0, 0], [-1, 0, 0]),
        # ([0, 0, 0], [0, 1, 0]),
        # ([0, 0, 0], [0, -1, 0]),
        # ([0, 0, 0], [0, 0, 1]),
        # ([0, 0, 0], [0, 0, -1]),
        ([0, 0, 0], [0, 0, 0]),
        ([1.0472, 0, 0], [0, 0, 0]), 
        ([-1.0472, 0, 0], [0, 0, 0]),
        ([0, 1.0472, 0], [0, 0, 0]),
        ([0, -1.0472, 0], [0, 0, 0]),
        ([0, 0, 1.0472], [0, 0, 0]),
        ([0, 0, -1.0472], [0, 0, 0])
    ]

    obs, _ = env.get_observations()

    # initialize variables to track current action
    action_iter = 0
    steps_per_action = 200
    action_ix = 0

    counter = 0

    while action_ix < len(goal_list):
        counter = counter + 1

        # get next action
        goal_orientation, goal_pos = goal_list[action_ix]

        t = counter / 60 # 1199
        goal_roll = 0.4 * np.sin(t * 1.4)
        goal_pitch = 0.55 * np.cos(t * 1.2)
        goal_yaw = np.sin(t)
        goal_orientation = [goal_roll, goal_pitch, goal_yaw]

        # run everything in inference mode
        with torch.inference_mode():
            des_ang_rpy = goal_orientation
            des_ang_quat = quat_from_euler_xyz(torch.Tensor([des_ang_rpy[0]]), torch.Tensor([des_ang_rpy[1]]), torch.Tensor([des_ang_rpy[2]]))
            env.unwrapped._goal[:] = des_ang_quat.to(env_cfg.sim.device)
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            true_pos = env.unwrapped._robot.data.root_pos_w[0].cpu().numpy()
            true_ang = obs[0, 5:9]
            true_ang_rpy = euler_xyz_from_quat(torch.unsqueeze(true_ang, 0))
            true_ang_rpy = np.array([true_ang_rpy[0][0].cpu().item(), true_ang_rpy[1][0].cpu().item(), true_ang_rpy[2][0].cpu().item()])
            true_ang_rpy = np.where(true_ang_rpy >= math.pi, true_ang_rpy - (2 * math.pi), true_ang_rpy)
            true_linvel = env.unwrapped._robot.data.root_lin_vel_b[0].cpu().numpy()
            true_angvel = env.unwrapped._robot.data.root_ang_vel_b[0].cpu().numpy()
            des_ang_rpy = euler_xyz_from_quat(torch.unsqueeze(des_ang_quat[0], 0))
            des_ang_rpy = np.array([des_ang_rpy[0][0].cpu().item(), des_ang_rpy[1][0].cpu().item(), des_ang_rpy[2][0].cpu().item()])
            des_ang_rpy = np.where(des_ang_rpy >= math.pi, des_ang_rpy - (2 * math.pi), des_ang_rpy)

            # 
            des_depth = goal_pos[2]
            log_row = {
                'des_depth': des_depth,
                'des_roll': des_ang_rpy[0],
                'des_pitch': des_ang_rpy[1],
                'des_yaw': des_ang_rpy[2],
                'true_x' : true_pos[0],
                'true_y' : true_pos[1],
                'true_z' : true_pos[2],
                'true_x_vel': true_linvel[0],
                'true_y_vel': true_linvel[1],
                'true_z_vel': true_linvel[2],
                'true_roll': true_ang_rpy[0],
                'true_pitch': true_ang_rpy[1],
                'true_yaw': true_ang_rpy[2],
                'true_roll_vel': true_angvel[0],
                'true_pitch_vel': true_angvel[1],
                'true_yaw_vel': true_angvel[2]
            }

            action_iter = action_iter + 1

            if (action_iter % steps_per_action) != 0:
                wandb.log(log_row)

            log_df = log_df._append(log_row, ignore_index=True)

            if counter > 2:
                ang_diff = quat_diff(ang_to_quat(np.array(log_df['true_roll']), np.array(log_df['true_pitch']), np.array(log_df['true_yaw'])), ang_to_quat(np.array(log_df['des_roll']), np.array(log_df['des_pitch']), np.array(log_df['des_yaw']))) 

                ang_diff = np.arccos(ang_diff[:, 0]) * 2  
                # MSE
                mse = np.mean(ang_diff ** 2)  

                print(f'counter {counter} | {len(goal_list) * steps_per_action} | MSE : {mse:.4f} rad^2      ', end='\r')

            action_ix = action_iter // steps_per_action

        
    # save logs dataframe
    log_df.to_csv(eval_csv_path)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()




            

            


            
