from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.terrains import TerrainImporter

from .commands import PathProgressVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_path_progress(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "base_velocity"
) -> torch.Tensor:
    """Adjust terrain levels using progress along the commanded linear-velocity path."""
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, PathProgressVelocityCommand):
        raise TypeError(
            f"Command term '{command_name}' must be PathProgressVelocityCommand, got {type(command).__name__}."
        )

    path_progress = command.path_progress[env_ids]
    commanded_path_length = command.commanded_path_length[env_ids]
    move_up = path_progress > terrain.cfg.terrain_generator.size[0] / 2
    move_down = (path_progress < commanded_path_length * 0.5) | env.termination_manager.terminated[env_ids]
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
