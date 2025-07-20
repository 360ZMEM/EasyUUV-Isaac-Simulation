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
parser.add_argument("--eval_name", type=str, default='wochao', help="Name of the eval run to store in wandb")
parser.add_argument("--custom_weights", type=str, default=None, help="Path to custom weights file")

# Eval parameters
parser.add_argument("--com_cob_offset", type=float, default=0.03, help="Distance of center of buoyancy from the center of mass along the X axis")
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
import numpy as np

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from asymmetric_noise_cfg import *

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

def generate_signal(duration=1.4, sample_rate=1000, amplitude = np.pi / 2, frequencies = [0.1, 0.2, 0.5, 1.0, 2.0, 3.5]):
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    signal = np.zeros_like(t)

    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)

    max_abs = np.max(np.abs(signal))
    if max_abs != 0:
        signal = signal / max_abs

    signal *= amplitude
    return t, signal

_, signal1 = generate_signal(amplitude=1.1, frequencies = [-0.1, 0.2, 0.5, -1.0, 2.0, 3.5]) 
_, signal2 = generate_signal(amplitude=1.35, frequencies = [-0.1, 0.2, 0.4, 0.8, 1.6, -3.2]) 
_, signal3 = generate_signal(amplitude=0.95, frequencies = [0.15, 0.3, 0.5, -0.9, 1.8, -3]) 

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )

    env_cfg.domain_randomization.use_custom_randomization = False
    # env_cfg.com_to_cob_offset[0] += args_cli.com_cob_offset
    env_cfg.volume = 0.0228

    env_cfg.use_boundaries = False
    env_cfg.cap_episode_length = False
    env_cfg.episode_length_before_reset = 0

    env_cfg.goal_spawn_radius = 0

    env_cfg.eval_mode = True


    # 0.1648 0.1448 0.2343
    # 0.1143 0.1571 0.1969

    env_cfg.control_method = 'Ssurface'
    env_cfg.s_ratio = 4.9
    env_cfg.self_adapt = True

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


    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load('/home/zmem063/isaaclab/logs/rsl_rl/warpauv_direct/SSA/model_500.pt')
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create dir to save logs into
    save_path = os.path.join("source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint[:-3] + "_play")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"[INFO]: Saving results into: {save_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    print('Policy export OK!')


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()




            

            


            
