#!/usr/bin/env python3
"""汇总一次运动策略训练的 TensorBoard 信号。"""

from __future__ import annotations

import argparse
import statistics

from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer


DEFAULT_TAGS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Curriculum/terrain_levels",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Policy/mean_noise_std",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/track_ang_vel_z_exp",
    "Episode_Reward/undesired_contacts",
    "Episode_Reward/foot_slippage",
    "Episode_Reward/contact_forces",
    "Episode_Reward/action_rate",
    "Episode_Reward/joint_acceleration",
    "Episode_Reward/joint_torques",
    "Episode_Reward/joint_position",
    "Episode_Reward/joint_position_limits",
    "Episode_Reward/joint_velocity_limits",
    "Episode_Reward/joint_torque_limits",
    "Episode_Reward/linear_velocity",
    "Episode_Reward/angular_velocity",
    "Episode_Reward/air_time_variance",
    "Episode_Termination/base_contact",
    "Episode_Termination/bad_orientation",
    "Episode_Termination/time_out",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdir", help="TensorBoard 训练目录")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[0, 100, 300, 500, 800, 1000, 1200, 1300],
        help="需要采样的训练步；使用距离最近的已有标量。",
    )
    parser.add_argument("--last", type=int, default=100, help="计算末尾均值所用的样本数。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multiplexer = EventMultiplexer(size_guidance={"scalars": 0}).AddRunsFromDirectory(args.logdir)
    multiplexer.Reload()
    runs = multiplexer.Runs()
    if not runs:
        raise ValueError(f"在 {args.logdir} 中没有找到 TensorBoard event 数据")

    for run_name, metadata in runs.items():
        scalar_tags = set(metadata.get("scalars", []))
        print(f"\nRUN {run_name} ({len(scalar_tags)} scalar tags)")
        for tag in DEFAULT_TAGS:
            if tag not in scalar_tags:
                continue
            events = multiplexer.Scalars(run_name, tag)
            trailing = events[-args.last :] if len(events) >= args.last else events
            sampled = []
            for requested_step in args.steps:
                event = min(events, key=lambda item: abs(item.step - requested_step))
                sampled.append(f"{event.step}:{event.value:.6g}")
            values = [event.value for event in events]
            print(tag)
            print(f"  points {' '.join(sampled)}")
            print(
                f"  n={len(events)} range=[{min(values):.6g},{max(values):.6g}] "
                f"last_mean={statistics.fmean(event.value for event in trailing):.6g} "
                f"latest={events[-1].step}:{events[-1].value:.6g}"
            )


if __name__ == "__main__":
    main()
