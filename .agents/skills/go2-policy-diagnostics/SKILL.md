---
name: go2-policy-diagnostics
description: 诊断 AME/RSL-RL Unitree Go2 运动策略检查点，结合训练快照、TensorBoard、无头回放、奖励分解、接触、关节限制和机身坐标系落脚点定位问题。当策略爬行、蹲伏、三足运动、落脚异常、执行器饱和、得分高但动作差，或需要比较不同检查点时使用。
---

# Go2 策略诊断

所有命令从仓库根目录运行。Python 使用 `uv`，不直接调用 `python` 或 `pip`。

## 一、先划清证据边界

1. 确认检查点和训练目录。
2. 读取该训练目录的 `params/env.yaml`、`params/agent.yaml` 及保存的 git diff。
3. 对照当前源码，列出训练后发生的改动。
4. 检查点只反映训练时的配置，不会继承后续源码修改。
5. 未达到计划总迭代次数的检查点只能视为中间状态。

禁止用当前奖励、终止条件或机器人配置解释旧检查点。

## 二、先看 TensorBoard

```bash
uv run --python .venv/bin/python \
  .agents/skills/go2-policy-diagnostics/scripts/analyze_tensorboard.py \
  logs/rsl_rl/go2_ame/<训练目录>
```

不能只看平均奖励，至少对照：

- `Train/mean_reward` 与 `Curriculum/terrain_levels`
- 速度误差与地形进度
- timeout 与失败终止
- 各 `Episode_Reward/*` 的量级
- `Policy/mean_noise_std`
- 关节限位奖励趋势

高奖励但地形等级低，通常表示任务或奖励错位。某奖励全程严格为 0，优先检查是否未生效或配置错误。

## 三、做可比的策略诊断

分别测试运动和站立：

```bash
uv run --python .venv/bin/python \
  .agents/skills/go2-policy-diagnostics/scripts/diagnose_go2_policy.py \
  --task AME-Go2-v0 \
  --checkpoint <checkpoint.pt> \
  --num_envs 64 --steps 600 --warmup_steps 50 \
  --command_x 0.6 \
  --output_dir diagnostics/<训练目录>_<检查点>_moving \
  --headless

uv run --python .venv/bin/python \
  .agents/skills/go2-policy-diagnostics/scripts/diagnose_go2_policy.py \
  --task AME-Go2-v0 \
  --checkpoint <checkpoint.pt> \
  --num_envs 32 --steps 400 --warmup_steps 50 \
  --command_x 0.0 \
  --output_dir diagnostics/<训练目录>_<检查点>_standing \
  --headless
```

比较检查点时必须保持 seed、命令、环境数量、步数和 warmup 一致。

## 四、解释诊断结果

- `summary.csv`
  - 判断蹲伏优先看 `base_to_mean_foot_height_m`，不要只看世界坐标 `base_height_m`。
  - 台阶、坑洞、地形原点都会改变世界坐标 z。
- `body_contacts.csv`
  - 分别检查头、base、hip、thigh、calf。
  - 稀疏接触的 p95 可能为 0，同时看接触时平均力和最大力。
- `foot_placements.csv`
  - 比较四足 touchdown 次数。
  - 比较前后足 x 对称性及左右足 `abs(y)` 对称性。
  - `abs(y)` 过小表示脚靠近身体中心线。
- `joints.csv`
  - 对照默认姿态检查平均位置和目标位置。
  - 检查软限位越界、跟踪误差、扭矩 p95/最大值和超过 80% 限制的比例。
  - `effort_limit_Nm` 接近 `1e9` 表示扭矩软限位奖励实际无法触发。
- `rewards.csv`
  - 比较加权后的每步奖励量级。
  - 跟踪奖励远大于所有正则项时，畸形但能跟踪速度的动作仍可能高分。
- `env0_timeseries.csv`
  - 只用于时序检查；env0 不能代替全环境统计。

## 五、先分类根因，再调奖励

1. **配置错误**：执行器限制错误、地图维度不匹配、检查点与配置错配、阈值未生效。
2. **奖励错位**：奖励和 timeout 很高，但姿态低、步态不对称或地形等级低。
3. **训练不足**：奖励、课程和动作仍在持续改善，尚未形成稳定策略。
4. **观测/动作错配**：地图、命令模式、顺序或 action scale 发生变化。
5. **阶段错配**：名称或恢复参数写 stage 2，但实际 `FINETUNE=False`。

先修复共享物理边界，再改奖励。奖励语义改变后必须重新训练，不要从不兼容检查点继续。

## 六、Go2 专项检查

- Go2 自定义执行器会按扭矩—转速曲线裁剪扭矩；同时要确保 articulation 暴露给奖励函数的 effort limit 是真实值。
- ANYmal 的力/扭矩阈值和权重必须按 Go2 物理量级一起缩放，不能只改阈值。
- 站立命令比例只会采样零速度，不会自动约束站立姿态。
- 严格 ANYmal 奖励若在 Go2 上收敛到三足蹲伏，应明确记录新增的形态适配项，例如默认关节姿态和腾空时间方差。
- torso 终止和非期望接触必须分别定义，不能仅按相似的 link 名称推断。

## 七、修改后验证

1. 编译改动的 Python 文件。
2. 用 1–2 个环境做 Isaac Sim 冒烟回放。
3. 在启动输出中确认观测维度和 active reward terms。
4. 确认 `joints.csv` 中 effort limit 已变为目标值。
5. 用旧失败检查点确认原本失效的奖励能够产生非零值。
6. 新开训练任务，不恢复不兼容检查点。
7. 在固定检查点重复相同的运动和站立测试。

## 八、维护调试记录

每次诊断或修改后都必须更新 [DEBUG_LOG.md](DEBUG_LOG.md)：

1. 写清检查点、训练配置快照和测试命令。
2. 记录修改前指标、假设、实际改动和验证结果。
3. 未重新训练的奖励改动必须标记为“待训练验证”，不能写成已改善。
4. 失败尝试也要记录，避免以后重复。
5. 新结果应追加新记录，不覆盖旧数据。

历史诊断和修改效果统一记录在 [DEBUG_LOG.md](DEBUG_LOG.md)。
