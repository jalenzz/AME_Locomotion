# 诊断 CSV 判读

按需阅读；标准回放产出这些文件。

## `summary.csv`

- 蹲伏优先看 `base_to_mean_foot_height_m`，不要只看世界坐标 `base_height_m`
- 台阶、坑洞、地形原点都会改变世界坐标 z

## `body_contacts.csv`

- 分别检查头、base、hip、thigh、calf
- 稀疏接触的 p95 可能为 0，同时看接触时平均力和最大力

## `foot_placements.csv`

- 比较四足 touchdown 次数
- 比较前后足 x 对称性及左右足 `abs(y)` 对称性
- `abs(y)` 过小表示脚靠近身体中心线

## `env_metrics.csv`

- 按环境比较 `mean_vx_mps`、`fraction_vx_below_25pct_cmd` 与 `terrain_level_mean/final`，识别只在困难地形停住而被均值掩盖的情况
- 速度跟踪必须同时检查实际 `wz`、`mean_abs_wz_error_radps`、累计转角、世界坐标前向进度和 `path_to_displacement_ratio`。机身坐标系 `vx` 接近命令不代表沿世界直线前进
- `command_wz=0` 时，持续实际 `wz`、大累计转角或很高的路径/净位移比 → 命令跟踪失败，不能被平均 `vx` 或平均 reward 掩盖
- 地形课程按机器人相对环境原点的净位移升级；绕圈或只在中心平台内运动不代表完成地形穿越
- `all_feet_air_fraction`、`simultaneous_touchdown_ge2/3_fraction`、`mean_contact_count` 区分交替步态与四足同步跳跃；`base_vz_std_mps`、`base_height_std_m` 识别非同步竖直弹跳
- `terrain_type_name` 按 terrain generator 列映射，不能只凭 terrain type id 猜测

## `task_verdict.csv`

- 先验收是否完成命令对应任务，不得只看站立、平均 reward 或机身 `vx`
- 固定直行：世界进度/命令距离比、实际 yaw rate 误差、路径/净位移比；逐环境报通过/失败原因
- 目标困难地形上的任务失败必须直接报告，不能用混合均值、简单地形成功或物理健康指标掩盖

## `joints.csv`

- 对照默认姿态检查平均位置和目标位置
- 检查软限位越界、跟踪误差、扭矩 p95/最大值和超过 80% 限制的比例
- `effort_limit_Nm` 接近 `1e9` → 扭矩软限位奖励实际无法触发

## `rewards.csv`

- 比较加权后每步奖励量级
- 跟踪奖励远大于所有正则项时，畸形但能跟踪速度的动作仍可能高分

## `env0_timeseries.csv`

- 只用于时序检查；env0 不能代替全环境统计
