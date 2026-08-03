# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the Unitree Go2 AME policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_go2_symmetric_states"]


# Policy and critic have the same term order during stage 1:
# base linear/angular velocity, projected gravity, velocity command,
# joint position/velocity, previous action, and the 26 x 16 x 3 map scan.
_PROPRIO_DIM = 48
_MAP_LENGTH = 26
_MAP_WIDTH = 16
_MAP_COORD_DIM = 3


@torch.no_grad()
def compute_go2_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return original and sagittal-plane-mirrored Go2 states.

    The transformation is an involution and preserves the AME observation
    order.  It swaps left/right legs, reflects lateral vector components, and
    mirrors the yaw-aligned terrain scan along its lateral axis.
    """

    del env  # The Go2 observation layout is fixed by the task configuration.

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group_name in ("policy", "critic"):
            if group_name not in obs.keys():
                continue
            obs_aug[group_name][:batch_size] = obs[group_name]
            obs_aug[group_name][batch_size:] = _mirror_observation_left_right(obs[group_name])
    else:
        obs_aug = None

    if actions is not None:
        actions_aug = torch.empty(
            (actions.shape[0] * 2, actions.shape[1]), dtype=actions.dtype, device=actions.device
        )
        actions_aug[: actions.shape[0]] = actions
        actions_aug[actions.shape[0] :] = _mirror_joint_data_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _mirror_observation_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a concatenated Go2 AME observation across the sagittal plane."""

    expected_dim = _PROPRIO_DIM + _MAP_LENGTH * _MAP_WIDTH * _MAP_COORD_DIM
    if obs.shape[-1] != expected_dim:
        raise ValueError(f"Expected Go2 observation dimension {expected_dim}, got {obs.shape[-1]}")

    mirrored = obs.clone()
    device = obs.device
    dtype = obs.dtype

    # Linear vectors reflect y; angular vectors are axial and reflect x/z.
    mirrored[:, 0:3] *= torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype)
    mirrored[:, 3:6] *= torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype)
    mirrored[:, 6:9] *= torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype)
    mirrored[:, 9:12] *= torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)

    mirrored[:, 12:24] = _mirror_joint_data_left_right(obs[:, 12:24])
    mirrored[:, 24:36] = _mirror_joint_data_left_right(obs[:, 24:36])
    mirrored[:, 36:48] = _mirror_joint_data_left_right(obs[:, 36:48])

    map_scan = obs[:, _PROPRIO_DIM:].reshape(
        -1, _MAP_WIDTH, _MAP_LENGTH, _MAP_COORD_DIM
    )
    map_mirrored = map_scan.flip(dims=[1]).clone()
    map_mirrored[..., 1] *= -1.0
    mirrored[:, _PROPRIO_DIM:] = map_mirrored.reshape(obs.shape[0], -1)

    return mirrored


def _mirror_joint_data_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap Go2 left/right joints and reflect hip-abduction coordinates."""

    if joint_data.shape[-1] != 12:
        raise ValueError(f"Expected 12 Go2 joint values, got {joint_data.shape[-1]}")

    # Isaac articulation order is grouped by joint type:
    # FL, FR, RL, RR hip; FL, FR, RL, RR thigh; FL, FR, RL, RR calf.
    mirrored = joint_data[..., [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]].clone()
    mirrored[..., 0:4] *= -1.0
    return mirrored
