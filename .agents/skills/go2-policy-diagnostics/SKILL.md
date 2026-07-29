---
name: go2-policy-diagnostics
description: 诊断 AME/RSL-RL Unitree Go2 运动策略检查点，结合训练快照、TensorBoard、无头回放、奖励分解、接触、关节限制和机身坐标系落脚点定位根因，并在用户确认后实施根因驱动的修改；也可在确认后通过 SSH 操作 4090 上 tmux session ame 的 window 0 启停训练。当策略爬行、蹲伏、三足运动、落脚异常、执行器饱和、得分高但动作差，或需要比较不同检查点时使用。
---

# Go2 策略诊断

所有命令从仓库根目录运行。Python 使用 `uv`，不直接调用 `python` 或 `pip`。

## 零、先同步训练结果

每次诊断前先从 4090 训练服务器增量同步 `logs/`：

```bash
rsync -avz --progress \
  4090:/home/root12/code/AME_Locomotion/logs/ \
  /home/matrix/code/AME_Locomotion/logs/
```

不添加 `--delete`。同步失败时，如果目标 run 已在本地，可继续诊断，但必须说明本地结果的新鲜度未确认；不得默默把旧日志当作最新结果。

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
- `env_metrics.csv`
  - 按环境比较 `mean_vx_mps`、`fraction_vx_below_25pct_cmd` 与 `terrain_level_mean/final`，识别只在困难地形停住而被均值掩盖的情况。
  - `all_feet_air_fraction`、`simultaneous_touchdown_ge2/3_fraction` 和 `mean_contact_count` 用于区分正常交替步态与四足同步跳跃；同时看 `base_vz_std_mps`、`base_height_std_m` 识别非同步的竖直弹跳。
  - `terrain_type_name` 按 terrain generator 的列映射输出，不能只凭 terrain type id 猜测地形类别。
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

先修复共享物理边界，再改奖励。以解决问题的预期有效性为首要标准，不为了最小 diff 或单变量而拆散同一机制下彼此依赖的改动；同时不捎带与根因无关的调参。奖励语义改变后必须重新训练，不要从不兼容检查点继续。

## 六、Go2 专项检查

- Go2 自定义执行器会按扭矩—转速曲线裁剪扭矩；同时要确保 articulation 暴露给奖励函数的 effort limit 是真实值。
- ANYmal 的力/扭矩阈值和权重必须按 Go2 物理量级一起缩放，不能只改阈值。
- 站立命令比例只会采样零速度，不会自动约束站立姿态。
- 严格 ANYmal 奖励若在 Go2 上收敛到三足蹲伏，应明确记录新增的形态适配项，例如默认关节姿态和腾空时间方差。
- torso 终止和非期望接触必须分别定义，不能仅按相似的 link 名称推断。

## 七、修改前的根因分析与用户确认（强制）

完成同步、训练曲线、配置/论文/官方实现对照和无头诊断后，必须先汇总全部证据，再决定是否修改。不得边测边凭单个指标打补丁。

提交给用户的修改建议只保留以下内容：

1. **现象与证据**：列出能复现问题的最少指标，并标明来自哪个 run/checkpoint、配置或源码。
2. **根因判断**：区分已确认根因、最可能根因和仍待验证的假设；说明排除了哪些近似解释。
3. **根因→修改映射**：每项修改必须说明它改变的机制、涉及的文件/配置和预期可观测结果。允许一次修改一组共同指向同一根因的改动，不以“最小 diff”为目标。
4. **不修改项**：说明为什么保留关键现有机制（例如课程变量、坑深、已有奖励），避免把无关参数一起调掉。

禁止以下补丁式做法：

- 为压低某个指标临时增加命令门控、阈值分支或只对失败地形生效的奖励；
- 没有奖励语义或官方/论文依据就堆新奖励、复制一套抽象奖励配置或任意调系数；
- 用改变地形深度、终止条件或统计口径来掩盖策略没有完成任务；
- 在没有先确认共享物理、课程、观测/动作和阶段配置的情况下直接调奖励。

分析结束时必须停下来向用户提问，明确列出拟修改项，并询问是否按该方案实施。此确认前不得编辑源码或配置、在 4090 tmux 启停训练、提交 commit 或 push。用户提出异议或新增证据时，重新更新“根因→修改映射”并再次询问；不能把沉默视为同意。

用户确认后才实施修改，并进入下一节验证流程。若只是诊断而不需要改代码，也要先给出根因结论和建议，等待用户决定是否实施。需要重新训练时，另走“在 4090 的 tmux 中启停训练”确认流程，不得把改码批准自动当作开训批准。

## 八、修改后验证

1. 编译改动的 Python 文件。
2. 用 1–2 个环境做 Isaac Sim 冒烟回放。
3. 在启动输出中确认观测维度和 active reward terms。
4. 确认 `joints.csv` 中 effort limit 已变为目标值。
5. 用旧失败检查点确认原本失效的奖励能够产生非零值。
6. 需要重新训练时，走下一节的 4090 tmux 启停流程；不恢复不兼容检查点。
7. 在固定检查点重复相同的运动和站立测试。

## 九、在 4090 的 tmux 中启停训练（强制确认）

训练在远程 `4090` 的 tmux `ame:0`（目录 `/home/root12/code/AME_Locomotion`）。只读查看（`capture-pane` / `pgrep`）无需确认；`send-keys`、停训、开训必须先问用户，改码批准不等于开训批准。不要动其他 window。

获批后默认：先对 `ame:0.0` 发一次 `C-c` 停训，确认退出后再 `bash run_train.sh`；先确认代码已在 4090，开训后回报新 run。

## 十、维护调试记录

用 [DEBUG_LOG.md](DEBUG_LOG.md) 记录训练—诊断—修改闭环：

1. 每个小节对应一个被诊断的训练 run，标题为 `<run> / <checkpoint> — <一句话结果>`。run 是主标识；commit 写在“训练”中。
2. 只在出现新训练 run 时开新小节。对照论文、研究外部实现、追加诊断和修复决策都并入当前小节。
3. 每节只保留四项：
   - **训练**：本轮验证的 commit/关键改动及迭代进度。
   - **结果**：能证明或否定假设的最少指标。
   - **判断**：已确认的改善、仍存问题和根因。
   - **下一步**：选定的有效修复及状态（待训练/已验证/放弃）。修复可以包含一组共同指向同一根因的改动。
4. 标准测试条件只在文件开头写一次；单节只记录偏离标准的条件。
5. 不记录工具/权限失败、无关工作区变更、完整命令、全量配置和未选中的方案。只记录会影响后续决策的失败实验。
6. 未有新检查点的改动必须写“待训练”，不得宣称已改善。

7. 在用户确认前，当前小节的“下一步”只能写拟议方案和“待确认”；确认后才写“已修改/待验证”。验证完成后补充结果。不要把未经批准的方案写成已实施。

## 十一、提交与推送确认（强制）

修改并完成编译、冒烟和必要的固定检查点验证后，先向用户报告：改了什么、验证结果、仍有哪些不确定性，以及是否需要重新训练。随后单独询问：**是否 commit？是否 push？**

在获得明确授权前，不得执行 `git commit` 或 `git push`。两者分别遵守授权：用户只批准 commit 时不得 push；用户批准 push 时先完成并报告 commit，再推送当前分支。推送前检查目标分支和远端，禁止顺带提交用户无关的 dirty 文件。
