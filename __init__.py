# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .easyuuv_env import EasyUUVEnv, EasyUUVEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="EasyUUV-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.isaac-auv-env-new:EasyUUVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EasyUUVEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.EasyUUVPPORunnerCfg
    },
)