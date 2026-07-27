"""AME locomotion task for Unitree Go2."""
import gymnasium as gym
from ame_locomotion.tasks.manager_based.ame_locomotion import agents

gym.register(
    id="AME-Go2-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:Go2RoughEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:Go2AMEPPORunnerCfg"},
)
gym.register(
    id="AME-Go2-Play-v0", entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.velocity_env_cfg_go2:Go2RoughEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.ame_rsl_rl_ppo_cfg:Go2AMEPPORunnerCfg"},
)
