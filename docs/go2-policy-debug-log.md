# Go2 策略调试

项目级闭环记录（不属于 skill 指令）。写法见 `.agents/skills/go2-policy-diagnostics/SKILL.md` 第八节。

每节对应一次训练闭环：训练 run 是主标识，commit 是该 run 验证的输入。标准回放为运动 `vx=0.6`、64 envs、600 steps 和站立 `vx=0`、32 envs、400 steps，均 warmup 50、seed 42。

## 2026-07-28_12-47-52 / model_1300 — 头部爬行

- **训练**：`e3e5580`，1300/15000；基本沿用 ANYmal-D 奖励和接触定义。
- **结果**：`vx=0.432 m/s`，倾角 0.327 rad，`Head_lower` 接触 19.1%；FL/FR/RL/RR 足接触 7.0%/76.4%/2.3%/22.7%。
- **判断**：策略用头部和少数足支撑换取速度；仅终止 base 接触无法阻止该行为。
- **下一步**：`e73978c` 扩展非期望接触并适配 Go2 地图、命令和尺度；已由 `2026-07-28_16-14-51` 验证：头部爬行消失，但转为三足蹲伏。

## 2026-07-28_16-14-51 / model_1300 — 三足蹲伏

- **训练**：`e73978c`，1300/15000；扩展头/腿接触，足力阈值改为 100 N，站立命令 5%。
- **结果**：头部接触约 1%，但运动/站立机身到足端高仅 0.164/0.153 m；RL 接触 8.1%/2.3%；训练 reward 147、timeout 99.6%，terrain level 0.562。
- **判断**：高分与合理步态错位。`joint_effort_limits=1e9 Nm` 使扭矩软限位奖励失效；100 N 足力仍用 700 N 的权重量级，贡献过小。
- **下一步**：`ac5b64d` 设 effort limit 23.4 Nm，按 Go2 量级放大 torque/足力惩罚，加入 `joint_position=-0.2` 和 `air_time_variance=-1`；已由 `2026-07-28_18-16-13` 验证。

## 2026-07-28_18-16-13 / model_3000 — 高度恢复，RL 仍弃用

- **训练**：`539defa`（行为改动 `ac5b64d`），3000/15000；训练最终进行到 10595。
- **结果**：运动 `vx=0.489/0.6 m/s`，机身到足端高 0.300 m，FL/FR/RL/RR 接触 81.3%/65.0%/1.9%/75.6%，RL touchdown 101，其他足约 2800；站立高度 0.304 m，RL 接触 0%。effort limit 为 23.4 Nm，扭矩饱和和软限位越界均极少。iteration 3000 的 reward 145.9、terrain level 0.573；末尾 terrain level 仍约 0.563。
- **判断**：`ac5b64d` 修复了蹲伏和执行器边界，未修复三足。不是扭矩/限位问题，而是姿态正则相对跟踪奖励仍太弱：站立时跟踪约 +0.160/步，`joint_position` -0.00556，`air_time_variance` -0.000122。Unitree 官方 Go2 也没有直接“四足必须接触”项；它依靠更强的默认姿态（-0.7，零速×5）、flat orientation、更强的 action/limit 正则以及更低的跟踪上限，且任务主要是平地。因此目前没有证据必须新增 contact/current-air-time 奖励。
- **下一步**：不加直接 contact/current-air-time 奖励；按官方 Go2 的姿态机制，将 `joint_position` 从 -0.2 调到 -0.7，并加入 `flat_orientation_l2=-2.5`。两项分别约束默认腿型和三足站立所需的机身倾斜，共同针对同一姿态根因；其他 action/limit/跟踪权重不变。旧三足检查点冒烟确认两项已激活且非零。状态：**已修改/待训练**。

## 2026-07-29_10-19-50 / model_3400 — 曲线进入平台，站立在 2100 后退化

- **训练**：`ce453a3`，3400/15000；`joint_position=-0.7`、`flat_orientation=-2.5`，2048 envs。
- **结果**：model 2100→3000→3400：站立 RL 接触 85.8%→65.0%→42.4%，全时段平均支撑力 13.36→10.51→6.39 N。3400 运动四足 touchdown 仍对称（2972–3003），`vx=0.466/0.6 m/s`。最近 1000 轮 reward 斜率近 0，terrain level 在 2000–2199 达到窗口均值 0.592 后回到约 0.53–0.56，XY 速度误差稳定在约 0.20。按环境看，stepping stones 的 mean `vx=0.057–0.215`、gaps 为 `0.078–0.094`，约 60–97% 时间低于命令 25%；普通地形同时 touchdown≥2 仅约 1.6%，但 boxes 上 `base_vz` 标准差约 0.25 m/s。
- **判断**：训练约在 2000–2500 后进入平台；问题不只是站立漂移，还包括困难地形上的“保持三足支撑而停止前进”和普通崎岖地形上的非同步竖直弹跳。`GO2_ROUGH_TERRAINS_CFG` 的 stepping stones/gaps 坑深固定为 -2.0 m，且停住仍保有约 0.70 的指数跟踪值、timeout 不触发终止惩罚，形成了风险规避局部最优。`model_2100` 是已测检查点中更好的行为候选，最新模型不是最优模型。
- **下一步**：保持 `gap_depth=-2.0` 及现有水平课程范围不变；新训练收紧速度跟踪核（`std=0.5`），增强现有竖直速度/动作平滑惩罚，保留 `air_time_variance` 原项。状态：**已修改/待训练**。

## 2026-07-29_15-03-39 / model_2600 — 困难地形和四足站立改善，课程判定误伤转弯

- **训练**：`0472aee`，诊断时 2668/15000、仍在运行；线速度跟踪 `std=0.5`，增强 action rate、关节加速度和竖直速度惩罚，2048 envs。
- **结果**：model 2600 运动 `vx=0.533/0.6 m/s`、机身到足端高 0.289 m、四足 touchdown 2538–2612；站立四足接触均为 99.9–100%、高度 0.304 m。相对上一 run 的 model 3400，stepping stones/gaps 平均 `vx` 从 0.136/0.086 提至 0.237/0.405，停滞环境从 12/64 降至 5/64。训练 terrain level 从约 0.90 回落到 0.73；完整 episode 对照中，直行使 53/64 环境升级，而 `vx=0.6,wz=0.5` 在速度跟踪正常时仍使 60/64 降级、0 个升级。
- **判断**：本轮奖励调整有效，未再出现爬行或弃用单足，model 2600 是当前已测最佳检查点；stepping stones/gaps 仍是主要失败地形。terrain level 回落不代表策略整体退化，而是官方净位移课程与持续转向命令不相容，并会让正确转弯的环境降级、减少高难度暴露。
- **下一步**：保留当前奖励、地形范围和终止条件；Go2 课程已改为累计实际速度在命令方向上的投影，失败终止仍降级。model 2600 完整 episode 复测中，直行 61/64 升级、2 个降级、1 个不变，`vx=0.6,wz=0.5` 转弯 64/64 升级。修复已于 `2026-07-29_18-00-54` 从 model 3000 恢复训练，目标仍为总计 15000 轮；启动输出确认新进度指标已生效。状态：**已修改/已验证/训练中**。

## 2026-07-29_18-00-54 / model_14999 — 速度稳定但 RL 后腿再次固定腾空

- **训练**：commit `ab5c0b7`，从 `2026-07-29_15-03-39/model_3000.pt` 恢复，完成 `15000/15000`；`FINETUNE=False`，站立关节位置/速度项仍为 `null`。
- **结果**：最终 mean reward `131.14`、episode length `987.15/1000`、terrain level `5.99`、timeout `0.9766`；训练后运动回放 `vx=0.563/0.6 m/s`，机身到足端高 `0.299 m`，FL/FR/RL/RR 接触 `80.5%/55.9%/4.5%/74.2%`；站立回放四足接触 `100%/100%/0%/100%`。effort limit `23.4 Nm`，软限位越界和普遍扭矩饱和均未出现。`model_14000` 在 `AME-Go2-Play-v0`、固定 `vx=0.6, vy=0, wz=0` 下从中心前进约 `0.72 m` 后开始转圈：30 秒平均实际 `wz=0.719 rad/s`、94.3% 时间 `|wz error|>0.3`、累计转角 `21.6 rad`、路径 `17.27 m` 但净位移仅约 `0.77 m`，路径/位移比 `22.34`；同时机身坐标系 `vx=0.564 m/s`、线速度奖励 `+0.0913/步`，角速度跟踪在明显失败时仍有 `+0.0352/步`，RL 接触仅 `0.8%`。训练任务 256 环境按地形分组还出现 steppingstones 环境 `vx≈0.088–0.102`、`82%` 时间低于命令 25%，以及 gaps 环境 `vx=0.083`、同样 `82%` 停滞。
- **判断**：训练完成且混合地形均值稳定，但 RL 后腿仍基本不着地。平台边缘转圈现已确认为命令跟踪失败：线速度奖励使用机身坐标系速度，使转圈时仍能接近满额；`track_ang_vel_z_exp(std=1.0)` 对约 `0.7 rad/s` 的错误仍给出过半角速度奖励，因此“留在安全平台绕圈”优于进入梅花桩承担速度下降风险。`a68970b` 又把机身前向速度累计为课程进度，使未离开平台的绕圈也能升级；此前诊断漏统计实际 `wz`、世界轨迹和累计转角，错误地让平均 `vx` 掩盖了该失败。
- **下一步**：角速度跟踪 `std` 已从 `1.0` 收紧到与 G1 一致的 `0.25`，旧失败检查点持续转圈阶段该奖励中位数约 `0.00006/步`。删除 `PathProgressVelocityCommand`，恢复 G1 使用的 heading command 与标准净位移地形课程；2 env、1 iteration 冒烟确认 `UniformVelocityCommand`、`heading_command=true`、`terrain_levels_vel` 生效。奖励和命令语义已改变，将从头训练，不恢复旧 checkpoint。状态：**已修改/已验证/待训练**。

## 2026-07-31_18-13-02 / model_16600 — 已能离开平台，但缺失运动状态且仍固定弃腿

- **训练**：commit `358545a`，16600/30000；1024 envs，Go2 gaps/stepping stones 中心平台 `1.5 m`，加入 bounded current-air-time penalty 和 touchdown reward。
- **结果**：`hf_gaps` 0/12，通过前进 `0.81 m`，低速比例 `80.7%`，平均 `1.87 s` 离开中心平台且 `87%` 时间位于平台外；`hf_steppingstones` 4/13，通过前进 `1.58 m`。运动 FL/FR/RL/RR 接触率为 `85.1%/58.1%/17.7%/77.3%`；站立为 `99.8%/92.6%/0.2%/99.9%`。RL current air time 均值 `0.99 s`、p95 `6.22 s`、最大 `11.56 s`。前方扫描在 gaps 中稳定出现 `-1.2 m`，关节扭矩饱和仍很少。
- **判断**：高度图能够感知 gap，平台大小和执行器不是当前主因。AME 论文要求 actor/critic 均观测 base linear velocity，但本 run actor 缺失该 3 维输入（actor/critic proprio `45/48`），使瞬时机器人坐标系地图下的停滞、打滑和正常前进部分不可辨。PPO 未启用 symmetry，且 long-air penalty 在总腾空 `1.0 s` 后饱和，允许训练随机固定弃用 RL 或 RR；touchdown 项对已弃用腿没有持续恢复信号。
- **下一步**：actor 恢复无噪声 `base_lin_vel`，使 stage-1 actor/critic 观测均为 `1296`；启用 Go2 左右镜像 PPO augmentation（地图、向量、关节和动作一致反射）；long-air 免费 swing 仍为 `0.5 s`，但 excess 上限由 `0.5 s` 延长到 `1.5 s`，touchdown 改用 Isaac Lab `compute_first_contact`。编译、镜像 involution/TensorDict 测试和 16 env、1 iteration 冒烟均通过，启动输出确认 symmetry metric、18 个 reward terms 及 actor/critic 48 维 proprio 生效。观测和奖励语义不兼容旧 checkpoint，将从头训练。状态：**已修改/已验证/待训练**。

## 2026-08-03_10-32-17 / iteration_0 — 新配置启动

- **训练**：commit `43b7c1f`，从头启动 `30000` iterations、1024 envs；启用 Go2 左右镜像 augmentation，`num_mini_batches=8` 保持增广后的峰值编码 batch 与旧配置相当。
- **结果**：iteration 0 完成，完整 iteration 用时约 28.9 s；actor/critic 均为 1296 维，`Mean symmetry loss=0.0014`，未发生 OOM 或启动错误。
- **判断**：训练进程和新配置均正常；尚无新 checkpoint，不能据此判断 gaps/steppingstones 行为已改善。
- **下一步**：训练继续运行；出现 `model_2000.pt` 或更早可用 checkpoint 后，按固定 moving/standing 条件复测四脚接触、long-air、平台外进度和 terrain-specific pass/fail。状态：**已修改/已验证/训练中**。
