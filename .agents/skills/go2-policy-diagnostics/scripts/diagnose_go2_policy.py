#!/usr/bin/env python3
"""无头运行 Go2 检查点并输出运动诊断 CSV。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="AME-Go2-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--sample_stride", type=int, default=1)
parser.add_argument("--contact_threshold", type=float, default=1.0)
parser.add_argument("--command_x", type=float, default=0.6)
parser.add_argument("--command_y", type=float, default=0.0)
parser.add_argument("--command_yaw", type=float, default=0.0)
parser.add_argument("--max_mean_abs_wz_error", type=float, default=0.3)
parser.add_argument("--min_straight_progress_ratio", type=float, default=0.5)
parser.add_argument("--max_path_to_displacement_ratio", type=float, default=2.0)
parser.add_argument("--output_dir", type=str, default="diagnostics/go2_policy")
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")

args_cli.enable_cameras = False
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import ame_locomotion.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401


def _policy_actions(policy, obs):
    output = policy(obs)
    return output[0] if isinstance(output, tuple) else output


def _percentile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().reshape(-1), q).item())


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.float().reshape(-1)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p05": _percentile(values, 0.05),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": float(values.max().item()),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.debug_vis = False
    command_cfg.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
    command_cfg.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
    command_cfg.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    # Play configurations may add a visualization camera. Diagnostics are headless and
    # explicitly disable cameras, so remove it before constructing the environment.
    if hasattr(env_cfg.scene, "visualize_cam"):
        env_cfg.scene.visualize_cam = None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs_result = env.get_observations()
    obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    joint_names = list(robot.joint_names)
    body_names = list(contact_sensor.body_names)
    foot_ids = [i for i, name in enumerate(body_names) if name.endswith("_foot")]
    robot_foot_ids, robot_foot_names = robot.find_bodies(".*_foot")
    sensor_foot_names = [body_names[i] for i in foot_ids]
    if robot_foot_names != sensor_foot_names:
        raise ValueError(f"Robot/contact foot order mismatch: {robot_foot_names} != {sensor_foot_names}")

    print(f"[INFO] checkpoint: {checkpoint}")
    print(f"[INFO] joints ({len(joint_names)}): {joint_names}")
    print(f"[INFO] contact bodies ({len(body_names)}): {body_names}")

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    terrain_type_names = []
    if terrain_generator_cfg is not None:
        subterrain_names = list(terrain_generator_cfg.sub_terrains.keys())
        proportions = [float(cfg.proportion) for cfg in terrain_generator_cfg.sub_terrains.values()]
        proportion_sum = sum(proportions)
        cumulative = []
        running = 0.0
        for proportion in proportions:
            running += proportion / proportion_sum
            cumulative.append(running)
        for column in range(int(terrain_generator_cfg.num_cols)):
            ratio = column / terrain_generator_cfg.num_cols + 0.001
            subterrain_index = next(i for i, limit in enumerate(cumulative) if ratio < limit)
            terrain_type_names.append(subterrain_names[subterrain_index])
        print(f"[INFO] terrain type columns: {dict(enumerate(terrain_type_names))}")

    os.makedirs(args_cli.output_dir, exist_ok=True)
    timeseries_path = os.path.join(args_cli.output_dir, "env0_timeseries.csv")
    summary_path = os.path.join(args_cli.output_dir, "summary.csv")
    contacts_path = os.path.join(args_cli.output_dir, "body_contacts.csv")
    foot_placements_path = os.path.join(args_cli.output_dir, "foot_placements.csv")
    joints_path = os.path.join(args_cli.output_dir, "joints.csv")
    rewards_path = os.path.join(args_cli.output_dir, "rewards.csv")

    samples = {
        "base_height": [],
        "base_to_mean_foot": [],
        "base_to_lowest_foot": [],
        "roll": [],
        "pitch": [],
        "inclination": [],
        "lin_vel": [],
        "ang_vel": [],
        "root_pos": [],
        "yaw": [],
        "command": [],
        "rewards": [],
        "actions": [],
        "joint_pos": [],
        "joint_target": [],
        "joint_vel": [],
        "torque": [],
        "contact_force": [],
        "foot_pos_b": [],
        "foot_contacts": [],
        "touchdowns": [],
        "terrain_levels": [],
        "terrain_types": [],
        "dones": [],
    }
    reward_term_samples = {name: [] for name in env.unwrapped.reward_manager.active_terms}
    touchdown_pos_samples = [[] for _ in foot_ids]
    previous_foot_contacts = torch.zeros(
        robot.data.root_pos_w.shape[0], len(foot_ids), dtype=torch.bool, device=env.unwrapped.device
    )
    initial_root_pos = robot.data.root_pos_w.detach().clone()
    initial_yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2].detach().clone()

    header = [
        "step", "time_s", "reward", "done", "root_x_m", "root_y_m", "relative_x_m", "relative_y_m",
        "base_height_m", "roll_rad", "pitch_rad", "yaw_rad", "inclination_rad",
        "command_vx", "command_vy", "command_wz", "base_vx", "base_vy", "base_vz",
        "base_wx", "base_wy", "base_wz",
    ]
    for name in joint_names:
        header.extend((f"{name}.action", f"{name}.pos", f"{name}.target", f"{name}.vel", f"{name}.torque"))
    for name in body_names:
        header.append(f"{name}.contact_N")

    with open(timeseries_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)

        for step in range(args_cli.steps):
            with torch.inference_mode():
                actions = _policy_actions(policy, obs)
                obs, rewards, dones, _ = env.step(actions)

            forces = torch.linalg.norm(contact_sensor.data.net_forces_w, dim=-1)
            foot_contacts = forces[:, foot_ids] > args_cli.contact_threshold
            touchdowns = foot_contacts & ~previous_foot_contacts
            foot_pos_w = robot.data.body_pos_w[:, robot_foot_ids] - robot.data.root_pos_w.unsqueeze(1)
            foot_pos_b = quat_apply_inverse(
                robot.data.root_quat_w.unsqueeze(1).expand(-1, len(robot_foot_ids), -1).reshape(-1, 4),
                foot_pos_w.reshape(-1, 3),
            ).reshape(robot.data.root_pos_w.shape[0], len(robot_foot_ids), 3)
            roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
            inclination = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
            command = env.unwrapped.command_manager.get_command("base_velocity")

            if step >= args_cli.warmup_steps:
                foot_height = robot.data.body_pos_w[:, robot_foot_ids, 2]
                values = {
                    "base_height": robot.data.root_pos_w[:, 2],
                    "base_to_mean_foot": robot.data.root_pos_w[:, 2] - foot_height.mean(dim=1),
                    "base_to_lowest_foot": robot.data.root_pos_w[:, 2] - foot_height.min(dim=1).values,
                    "roll": roll,
                    "pitch": pitch,
                    "inclination": inclination,
                    "lin_vel": robot.data.root_lin_vel_b,
                    "ang_vel": robot.data.root_ang_vel_b,
                    "root_pos": robot.data.root_pos_w,
                    "yaw": yaw,
                    "command": command,
                    "rewards": rewards,
                    "actions": actions,
                    "joint_pos": robot.data.joint_pos,
                    "joint_target": robot.data.joint_pos_target,
                    "joint_vel": robot.data.joint_vel,
                    "torque": robot.data.applied_torque,
                    "contact_force": forces,
                    "foot_pos_b": foot_pos_b,
                    "foot_contacts": foot_contacts,
                    "touchdowns": touchdowns,
                    "terrain_levels": env.unwrapped.scene.terrain.terrain_levels,
                    "terrain_types": env.unwrapped.scene.terrain.terrain_types,
                    "dones": dones,
                }
                for name, value in values.items():
                    samples[name].append(value.detach().cpu())
                for foot_index in range(len(foot_ids)):
                    if touchdowns[:, foot_index].any():
                        touchdown_pos_samples[foot_index].append(
                            foot_pos_b[touchdowns[:, foot_index], foot_index].detach().cpu()
                        )
                for term_name in reward_term_samples:
                    term_cfg = env.unwrapped.reward_manager.get_term_cfg(term_name)
                    term_value = term_cfg.func(env.unwrapped, **term_cfg.params)
                    reward_term_samples[term_name].append(
                        (term_value * term_cfg.weight * env.unwrapped.step_dt).detach().cpu()
                    )

            previous_foot_contacts.copy_(foot_contacts)
            previous_foot_contacts[dones.bool()] = False

            if step % args_cli.sample_stride == 0:
                row = [
                    step,
                    step * env.unwrapped.step_dt,
                    float(rewards[0].item()),
                    int(dones[0].item()),
                    float(robot.data.root_pos_w[0, 0].item()),
                    float(robot.data.root_pos_w[0, 1].item()),
                    float((robot.data.root_pos_w[0, 0] - initial_root_pos[0, 0]).item()),
                    float((robot.data.root_pos_w[0, 1] - initial_root_pos[0, 1]).item()),
                    float(robot.data.root_pos_w[0, 2].item()),
                    float(roll[0].item()),
                    float(pitch[0].item()),
                    float(yaw[0].item()),
                    float(inclination[0].item()),
                    float(command[0, 0].item()),
                    float(command[0, 1].item()),
                    float(command[0, 2].item()),
                    float(robot.data.root_lin_vel_b[0, 0].item()),
                    float(robot.data.root_lin_vel_b[0, 1].item()),
                    float(robot.data.root_lin_vel_b[0, 2].item()),
                    float(robot.data.root_ang_vel_b[0, 0].item()),
                    float(robot.data.root_ang_vel_b[0, 1].item()),
                    float(robot.data.root_ang_vel_b[0, 2].item()),
                ]
                for joint_id in range(len(joint_names)):
                    row.extend(
                        (
                            float(actions[0, joint_id].item()),
                            float(robot.data.joint_pos[0, joint_id].item()),
                            float(robot.data.joint_pos_target[0, joint_id].item()),
                            float(robot.data.joint_vel[0, joint_id].item()),
                            float(robot.data.applied_torque[0, joint_id].item()),
                        )
                    )
                row.extend(float(value) for value in forces[0].tolist())
                writer.writerow(row)

    if not samples["base_height"]:
        raise ValueError("No samples collected: --warmup_steps must be smaller than --steps")

    stacked = {name: torch.stack(values) for name, values in samples.items()}
    summary_metrics = {
        "base_height_m": stacked["base_height"],
        "base_to_mean_foot_height_m": stacked["base_to_mean_foot"],
        "base_to_lowest_foot_height_m": stacked["base_to_lowest_foot"],
        "roll_rad": stacked["roll"],
        "pitch_rad": stacked["pitch"],
        "inclination_rad": stacked["inclination"],
        "base_vx_mps": stacked["lin_vel"][..., 0],
        "base_vy_mps": stacked["lin_vel"][..., 1],
        "base_vz_mps": stacked["lin_vel"][..., 2],
        "base_wz_radps": stacked["ang_vel"][..., 2],
        "abs_wz_tracking_error_radps": (
            stacked["ang_vel"][..., 2] - stacked["command"][..., 2]
        ).abs(),
        "reward_per_step": stacked["rewards"],
        "done_fraction": stacked["dones"].float(),
    }
    with open(summary_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["metric", "mean", "std", "p05", "p50", "p95", "max"])
        writer.writeheader()
        for metric, values in summary_metrics.items():
            writer.writerow({"metric": metric, **_stats(values)})

    # Per-environment metrics expose terrain-conditioned failures hidden by aggregate means.
    # In particular, a policy may preserve a good mean velocity while stopping on only the
    # harder terrain instances, or achieve similar touchdown counts with a hopping gait.
    foot_contacts = stacked["foot_contacts"].bool()  # [time, env, foot]
    touchdowns = stacked["touchdowns"].bool()
    velocity_x = stacked["lin_vel"][..., 0]
    angular_velocity_z = stacked["ang_vel"][..., 2]
    commands = stacked["command"]
    root_pos = stacked["root_pos"]
    terrain_levels = stacked["terrain_levels"].float()
    terrain_types = stacked["terrain_types"][0].long()
    command_x = float(args_cli.command_x)
    per_env_path = os.path.join(args_cli.output_dir, "env_metrics.csv")
    verdict_path = os.path.join(args_cli.output_dir, "task_verdict.csv")
    verdict_rows = []
    with open(per_env_path, "w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "env_id", "terrain_type_id", "terrain_type_name", "terrain_level_mean", "terrain_level_final",
            "mean_vx_mps", "p05_vx_mps", "mean_abs_vx_error_mps", "fraction_vx_below_25pct_cmd",
            "mean_wz_radps", "mean_abs_wz_error_radps", "p95_abs_wz_error_radps",
            "fraction_abs_wz_error_above_0_3", "net_heading_change_rad", "total_abs_turn_rad",
            "forward_progress_from_start_m", "lateral_displacement_from_start_m",
            "commanded_forward_distance_m", "straight_progress_ratio",
            "xy_path_length_m", "path_to_displacement_ratio",
            "mean_reward", "base_vz_std_mps", "base_vz_p95_abs_mps", "base_height_std_m",
            "mean_contact_count", "all_feet_air_fraction", "all_feet_contact_fraction",
            "simultaneous_touchdown_ge2_fraction", "simultaneous_touchdown_ge3_fraction",
            "FL_contact_fraction", "FR_contact_fraction", "RL_contact_fraction", "RR_contact_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for env_id in range(velocity_x.shape[1]):
            vx = velocity_x[:, env_id]
            wz = angular_velocity_z[:, env_id]
            command = commands[:, env_id]
            wz_error = wz - command[:, 2]
            vz = stacked["lin_vel"][:, env_id, 2]
            base_height = stacked["base_height"][:, env_id]
            contacts = foot_contacts[:, env_id]
            td = touchdowns[:, env_id]
            td_count = td.sum(dim=1).float()
            displacement_w = root_pos[-1, env_id, :2] - root_pos[0, env_id, :2]
            initial_heading = float(initial_yaw[env_id].cpu().item())
            initial_forward = torch.tensor([math.cos(initial_heading), math.sin(initial_heading)])
            initial_lateral = torch.tensor([-initial_forward[1], initial_forward[0]])
            path_steps = root_pos[1:, env_id, :2] - root_pos[:-1, env_id, :2]
            path_length = float(torch.linalg.norm(path_steps, dim=1).sum().item())
            displacement_norm = float(torch.linalg.norm(displacement_w).item())
            forward_progress = float(torch.dot(displacement_w, initial_forward).item())
            commanded_forward_distance = float((command[:-1, 0] * env.unwrapped.step_dt).sum().item())
            straight_command = bool(
                command[:, 1].abs().max().item() < 1.0e-3
                and command[:, 2].abs().max().item() < 1.0e-3
                and commanded_forward_distance > 1.0e-6
            )
            straight_progress_ratio = (
                forward_progress / commanded_forward_distance if straight_command else None
            )
            path_to_displacement_ratio = path_length / max(displacement_norm, 1.0e-6)
            mean_abs_wz_error = float(wz_error.abs().mean().item())
            failure_reasons = []
            if straight_command and straight_progress_ratio < args_cli.min_straight_progress_ratio:
                failure_reasons.append("insufficient_world_progress")
            if straight_command and mean_abs_wz_error > args_cli.max_mean_abs_wz_error:
                failure_reasons.append("unexpected_yaw_rate")
            if straight_command and path_to_displacement_ratio > args_cli.max_path_to_displacement_ratio:
                failure_reasons.append("looping_or_large_detour")
            terrain_type_name = (
                terrain_type_names[int(terrain_types[env_id].item())] if terrain_type_names else "unknown"
            )
            row = {
                "env_id": env_id,
                "terrain_type_id": int(terrain_types[env_id].item()),
                "terrain_type_name": terrain_type_name,
                "terrain_level_mean": float(terrain_levels[:, env_id].mean().item()),
                "terrain_level_final": float(terrain_levels[-1, env_id].item()),
                "mean_vx_mps": float(vx.mean().item()),
                "p05_vx_mps": _percentile(vx, 0.05),
                "mean_abs_vx_error_mps": float((vx - command_x).abs().mean().item()),
                "fraction_vx_below_25pct_cmd": float((vx < 0.25 * command_x).float().mean().item())
                if command_x > 0.0 else 0.0,
                "mean_wz_radps": float(wz.mean().item()),
                "mean_abs_wz_error_radps": mean_abs_wz_error,
                "p95_abs_wz_error_radps": _percentile(wz_error.abs(), 0.95),
                "fraction_abs_wz_error_above_0_3": float((wz_error.abs() > 0.3).float().mean().item()),
                "net_heading_change_rad": float((wz * env.unwrapped.step_dt).sum().item()),
                "total_abs_turn_rad": float((wz.abs() * env.unwrapped.step_dt).sum().item()),
                "forward_progress_from_start_m": forward_progress,
                "lateral_displacement_from_start_m": float(torch.dot(displacement_w, initial_lateral).item()),
                "commanded_forward_distance_m": commanded_forward_distance,
                "straight_progress_ratio": straight_progress_ratio,
                "xy_path_length_m": path_length,
                "path_to_displacement_ratio": path_to_displacement_ratio,
                "mean_reward": float(stacked["rewards"][:, env_id].mean().item()),
                "base_vz_std_mps": float(vz.std(unbiased=False).item()),
                "base_vz_p95_abs_mps": _percentile(vz.abs(), 0.95),
                "base_height_std_m": float(base_height.std(unbiased=False).item()),
                "mean_contact_count": float(contacts.float().sum(dim=1).mean().item()),
                "all_feet_air_fraction": float((contacts.sum(dim=1) == 0).float().mean().item()),
                "all_feet_contact_fraction": float((contacts.sum(dim=1) == len(foot_ids)).float().mean().item()),
                "simultaneous_touchdown_ge2_fraction": float((td_count >= 2).float().mean().item()),
                "simultaneous_touchdown_ge3_fraction": float((td_count >= 3).float().mean().item()),
            }
            for foot_index, foot_name in enumerate(sensor_foot_names):
                row[f"{foot_name.replace('_foot', '')}_contact_fraction"] = float(
                    contacts[:, foot_index].float().mean().item()
                )
            writer.writerow(row)
            verdict_rows.append(
                {
                    "env_id": env_id,
                    "terrain_type_name": terrain_type_name,
                    "task_evaluated": straight_command,
                    "task_passed": straight_command and not failure_reasons,
                    "failure_reasons": ";".join(failure_reasons),
                }
            )

    print(f"[RESULT] wrote per-environment metrics to {per_env_path}")
    with open(verdict_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["env_id", "terrain_type_name", "task_evaluated", "task_passed", "failure_reasons"],
        )
        writer.writeheader()
        writer.writerows(verdict_rows)
    evaluated_rows = [row for row in verdict_rows if row["task_evaluated"]]
    failed_rows = [row for row in evaluated_rows if not row["task_passed"]]
    if evaluated_rows:
        status = "FAIL" if failed_rows else "PASS"
        print(f"[{status}] straight-command task: {len(failed_rows)}/{len(evaluated_rows)} environments failed")
    else:
        print("[INFO] task verdict skipped: command is not fixed straight motion")
    print(f"[RESULT] wrote task verdict to {verdict_path}")

    with open(rewards_path, "w", newline="", encoding="utf-8") as stream:
        fieldnames = ["term", "mean_per_step", "std", "p05", "p50", "p95", "min", "max"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for term_name, term_samples in reward_term_samples.items():
            values = torch.stack(term_samples).float().reshape(-1)
            stats = _stats(values)
            writer.writerow(
                {
                    "term": term_name,
                    "mean_per_step": stats["mean"],
                    "std": stats["std"],
                    "p05": stats["p05"],
                    "p50": stats["p50"],
                    "p95": stats["p95"],
                    "min": float(values.min().item()),
                    "max": stats["max"],
                }
            )

    contact_force = stacked["contact_force"]
    with open(contacts_path, "w", newline="", encoding="utf-8") as stream:
        fieldnames = ["body", "contact_fraction", "mean_force_N", "mean_force_when_contact_N", "p95_force_N", "max_force_N"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for body_id, body_name in enumerate(body_names):
            values = contact_force[..., body_id].reshape(-1)
            mask = values > args_cli.contact_threshold
            writer.writerow(
                {
                    "body": body_name,
                    "contact_fraction": float(mask.float().mean().item()),
                    "mean_force_N": float(values.mean().item()),
                    "mean_force_when_contact_N": float(values[mask].mean().item()) if mask.any() else 0.0,
                    "p95_force_N": _percentile(values, 0.95),
                    "max_force_N": float(values.max().item()),
                }
            )

    foot_pos_b = stacked["foot_pos_b"]
    with open(foot_placements_path, "w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "foot", "contact_samples", "mean_x_b_m", "p05_x_b_m", "p50_x_b_m", "p95_x_b_m",
            "mean_y_b_m", "p05_y_b_m", "p50_y_b_m", "p95_y_b_m", "mean_z_b_m",
            "touchdown_samples", "touchdown_mean_x_b_m", "touchdown_p05_x_b_m",
            "touchdown_p50_x_b_m", "touchdown_p95_x_b_m", "touchdown_mean_y_b_m",
            "touchdown_p05_y_b_m", "touchdown_p50_y_b_m", "touchdown_p95_y_b_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for foot_index, sensor_body_id in enumerate(foot_ids):
            mask = contact_force[..., sensor_body_id] > args_cli.contact_threshold
            positions = foot_pos_b[..., foot_index, :][mask]
            row = {"foot": body_names[sensor_body_id], "contact_samples": int(positions.shape[0])}
            if positions.numel():
                row.update(
                    {
                        "mean_x_b_m": float(positions[:, 0].mean().item()),
                        "p05_x_b_m": _percentile(positions[:, 0], 0.05),
                        "p50_x_b_m": _percentile(positions[:, 0], 0.50),
                        "p95_x_b_m": _percentile(positions[:, 0], 0.95),
                        "mean_y_b_m": float(positions[:, 1].mean().item()),
                        "p05_y_b_m": _percentile(positions[:, 1], 0.05),
                        "p50_y_b_m": _percentile(positions[:, 1], 0.50),
                        "p95_y_b_m": _percentile(positions[:, 1], 0.95),
                        "mean_z_b_m": float(positions[:, 2].mean().item()),
                    }
                )
            if touchdown_pos_samples[foot_index]:
                touchdown_positions = torch.cat(touchdown_pos_samples[foot_index])
                row.update(
                    {
                        "touchdown_samples": int(touchdown_positions.shape[0]),
                        "touchdown_mean_x_b_m": float(touchdown_positions[:, 0].mean().item()),
                        "touchdown_p05_x_b_m": _percentile(touchdown_positions[:, 0], 0.05),
                        "touchdown_p50_x_b_m": _percentile(touchdown_positions[:, 0], 0.50),
                        "touchdown_p95_x_b_m": _percentile(touchdown_positions[:, 0], 0.95),
                        "touchdown_mean_y_b_m": float(touchdown_positions[:, 1].mean().item()),
                        "touchdown_p05_y_b_m": _percentile(touchdown_positions[:, 1], 0.05),
                        "touchdown_p50_y_b_m": _percentile(touchdown_positions[:, 1], 0.50),
                        "touchdown_p95_y_b_m": _percentile(touchdown_positions[:, 1], 0.95),
                    }
                )
            writer.writerow(row)

    joint_pos = stacked["joint_pos"]
    joint_target = stacked["joint_target"]
    torque = stacked["torque"]
    with open(joints_path, "w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "joint", "mean_action", "mean_pos_rad", "mean_target_rad", "mean_abs_tracking_error_rad",
            "soft_lower_rad", "soft_upper_rad", "soft_limit_violation_fraction",
            "soft_velocity_limit_radps", "effort_limit_Nm",
            "mean_abs_velocity_radps", "mean_abs_torque_Nm", "p95_abs_torque_Nm", "max_abs_torque_Nm",
            "torque_above_80pct_limit_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for joint_id, joint_name in enumerate(joint_names):
            abs_torque = torque[..., joint_id].abs().reshape(-1)
            positions = joint_pos[..., joint_id].reshape(-1)
            soft_lower = float(robot.data.soft_joint_pos_limits[0, joint_id, 0].item())
            soft_upper = float(robot.data.soft_joint_pos_limits[0, joint_id, 1].item())
            effort_limit = float(robot.data.joint_effort_limits[0, joint_id].item())
            writer.writerow(
                {
                    "joint": joint_name,
                    "mean_action": float(stacked["actions"][..., joint_id].mean().item()),
                    "mean_pos_rad": float(joint_pos[..., joint_id].mean().item()),
                    "mean_target_rad": float(joint_target[..., joint_id].mean().item()),
                    "mean_abs_tracking_error_rad": float(
                        (joint_target[..., joint_id] - joint_pos[..., joint_id]).abs().mean().item()
                    ),
                    "soft_lower_rad": soft_lower,
                    "soft_upper_rad": soft_upper,
                    "soft_limit_violation_fraction": float(
                        ((positions < soft_lower) | (positions > soft_upper)).float().mean().item()
                    ),
                    "soft_velocity_limit_radps": float(robot.data.soft_joint_vel_limits[0, joint_id].item()),
                    "effort_limit_Nm": effort_limit,
                    "mean_abs_velocity_radps": float(stacked["joint_vel"][..., joint_id].abs().mean().item()),
                    "mean_abs_torque_Nm": float(abs_torque.mean().item()),
                    "p95_abs_torque_Nm": _percentile(abs_torque, 0.95),
                    "max_abs_torque_Nm": float(abs_torque.max().item()),
                    "torque_above_80pct_limit_fraction": float(
                        (abs_torque > 0.8 * effort_limit).float().mean().item()
                    ),
                }
            )

    print("\n[RESULT] aggregate locomotion diagnostics")
    print(f"  base height: {_stats(stacked['base_height'])['mean']:.3f} m")
    print(f"  base-to-feet: {_stats(stacked['base_to_mean_foot'])['mean']:.3f} m")
    print(f"  |pitch|: {stacked['pitch'].abs().mean().item():.3f} rad")
    print(f"  inclination: {_stats(stacked['inclination'])['mean']:.3f} rad")
    print(f"  base vx: {_stats(stacked['lin_vel'][..., 0])['mean']:.3f} m/s (command {args_cli.command_x:.3f})")
    print(
        f"  |wz error|: {(stacked['ang_vel'][..., 2] - stacked['command'][..., 2]).abs().mean().item():.3f} "
        f"rad/s (command {args_cli.command_yaw:.3f})"
    )
    print(f"  done rate: {stacked['dones'].float().mean().item():.5f} per policy step")
    print("  foot contact fractions:")
    for body_id in foot_ids:
        duty = (contact_force[..., body_id] > args_cli.contact_threshold).float().mean().item()
        print(f"    {body_names[body_id]:16s} {duty:.3f}")
    print(f"[RESULT] wrote diagnostics to {args_cli.output_dir}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
