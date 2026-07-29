from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import (
    TerrainBasedPose2dCommand,
    TerrainBasedPose2dCommandCfg,
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

from isaaclab.utils import configclass


class PathProgressVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command that tracks progress along the commanded path."""

    def __init__(self, cfg: PathProgressVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.path_progress = torch.zeros(self.num_envs, device=self.device)
        self.commanded_path_length = torch.zeros(self.num_envs, device=self.device)
        self.metrics["path_progress"] = self.path_progress
        self.metrics["commanded_path_length"] = self.commanded_path_length

    def _update_metrics(self):
        super()._update_metrics()
        linear_command = self.vel_command_b[:, :2]
        command_speed = torch.linalg.vector_norm(linear_command, dim=-1)
        progress_speed = torch.sum(self.robot.data.root_lin_vel_b[:, :2] * linear_command, dim=-1)
        progress_speed /= command_speed.clamp_min(1.0e-6)
        progress_speed = torch.where(command_speed > 1.0e-6, progress_speed, 0.0)
        self.path_progress += progress_speed * self._env.step_dt
        self.commanded_path_length += command_speed * self._env.step_dt


@configclass
class PathProgressVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for velocity commands with path-progress tracking."""

    class_type: type = PathProgressVelocityCommand


class TimeLimitedTerrainBasedPose2dCommand(TerrainBasedPose2dCommand):
    """Command generator that generates pose commands containing a 3-D position, heading, and time-to-go.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.

    The command tensor includes the time remaining to reach the target as the last element.
    Shape: (num_envs, 5) -> [x, y, z, heading, time_to_go]
    """

    cfg: TimeLimitedTerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TimeLimitedTerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose and time-to-go in base frame. Shape is (num_envs, 5).

        Indices:
            0-2: Position command (x, y, z) in base frame.
            3: Heading command in base frame.
            4: Time to go.
        """
        return torch.cat([
            self.pos_command_b,
            self.heading_command_b.unsqueeze(1),
            self.time_left.unsqueeze(1)
        ], dim=1)


@configclass
class TimeLimitedTerrainBasedPose2dCommandCfg(TerrainBasedPose2dCommandCfg):

    class_type = TimeLimitedTerrainBasedPose2dCommand
