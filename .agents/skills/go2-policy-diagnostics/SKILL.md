---
name: go2-policy-diagnostics
description: 诊断 AME/RSL-RL Unitree Go2 运动策略检查点，结合训练快照、TensorBoard、无头回放、奖励分解、接触、关节限制和机身坐标系落脚点定位根因，并在用户确认后实施根因驱动的修改；也可在确认后通过 SSH 操作 4090 上 tmux session ame 的 window 0 启停训练。当策略爬行、蹲伏、三足运动、落脚异常、执行器饱和、得分高但动作差，或需要比较不同检查点时使用。
---

# Go2 策略诊断

所有命令从仓库根目录运行。Python 使用 `uv`，不直接调用 `python` 或 `pip`。

详细 CSV 判读见 [metrics.md](metrics.md)。闭环记录见 [docs/go2-policy-debug-log.md](../../../docs/go2-policy-debug-log.md)。

## 零、先同步训练结果

```bash
rsync -avz --progress \
  4090:/home/root12/code/AME_Locomotion/logs/ \
  /home/matrix/code/AME_Locomotion/logs/
```

不添加 `--delete`。同步失败且本地已有目标 run 时可继续，但必须说明新鲜度未确认。

## 一、先划清证据边界

1. 确认检查点和训练目录。
2. 读取该训练目录的 `params/env.yaml`、`params/agent.yaml` 及保存的 git diff。
3. 对照当前源码，列出训练后发生的改动。
4. 检查点只反映训练时配置，不继承后续源码修改；未跑满计划迭代只能视为中间状态。

禁止用当前奖励、终止条件或机器人配置解释旧检查点。

### 对照入口（改奖励前必读）

| 来源 | 路径 / 链接 | 看什么 |
|------|-------------|--------|
| AME 论文 | [`docs/ame-paper/main_arxiv.tex`](../../../docs/ame-paper/main_arxiv.tex)（[arXiv:2506.09588](https://arxiv.org/abs/2506.09588)） | `tab:rew_func`、stage-1/2 橙色项、地形与网络维 |
| 本仓库 Go2 配置 | `source/ame_locomotion/.../go2/velocity_env_cfg_go2.py` | 当前奖励、终止、接触、课程 |
| Isaac Lab Go2 | `.venv/.../isaaclab_tasks/.../velocity/config/go2/rough_env_cfg.py` 及其父类 `velocity_env_cfg.py` | 官方 Go2 量级与默认项 |
| Isaac Lab ANYmal-D | 同目录树下 `anymal_d/rough_env_cfg.py` | 论文 Table 2 的 ANYmal 基线在 Lab 中的对应 |
| Unitree RL Lab（若本地无克隆） | https://github.com/unitreerobotics/unitree_rl_lab | Go2 接触惩罚、姿态正则等形态适配参考 |

## 二、先看 TensorBoard

```bash
uv run --python .venv/bin/python \
  .agents/skills/go2-policy-diagnostics/scripts/analyze_tensorboard.py \
  logs/rsl_rl/go2_ame/<训练目录>
```

至少对照：`Train/mean_reward` 与 `Curriculum/terrain_levels`、速度误差与地形进度、timeout 与失败终止、各 `Episode_Reward/*` 量级、`Policy/mean_noise_std`、关节限位奖励趋势。

高奖励但地形等级低 → 任务或奖励错位。某奖励全程严格为 0 → 优先查未生效或配置错误。

## 三、做可比的策略诊断

标准条件（与 debug log 一致）：seed `42`；运动 `vx=0.6`、64 envs、600 steps；站立 `vx=0`、32 envs、400 steps；均 warmup 50。比较检查点时 seed、命令、envs、steps、warmup 必须一致。

```bash
uv run --python .venv/bin/python \
  .agents/skills/go2-policy-diagnostics/scripts/diagnose_go2_policy.py \
  --task AME-Go2-v0 \
  --checkpoint <checkpoint.pt> \
  --num_envs 64 --steps 600 --warmup_steps 50 \
  --seed 42 --command_x 0.6 \
  --output_dir diagnostics/<训练目录>_<检查点>_moving \
  --headless

uv run --python .venv/bin/python \
  .agents/skills/go2-policy-diagnostics/scripts/diagnose_go2_policy.py \
  --task AME-Go2-v0 \
  --checkpoint <checkpoint.pt> \
  --num_envs 32 --steps 400 --warmup_steps 50 \
  --seed 42 --command_x 0.0 \
  --output_dir diagnostics/<训练目录>_<检查点>_standing \
  --headless
```

用户报告特定任务/地形行为问题时，必须用相同 task、检查点、地形和固定命令额外复测；其他任务的多环境统计不能替代。可视化相机不得阻止无头诊断。

判读产出 CSV 时打开 [metrics.md](metrics.md)。

## 四、先分类根因，再调奖励

1. **配置错误**：执行器限制、地图维、检查点/配置错配、项未生效
2. **奖励错位**：高 reward/timeout，但姿态低、步态不对称或地形等级低
3. **训练不足**：曲线与动作仍在改善
4. **观测/动作错配**：地图、命令模式、顺序或 action scale 变化
5. **阶段错配**：名称写 stage 2 但 `FINETUNE=False`

先修共享物理边界，再改奖励。同一根因下彼此依赖的改动可一组提交；不捎带无关调参。奖励语义改变后必须重新训练，不从不兼容检查点继续。

## 五、Go2 专项检查

- 自定义执行器按扭矩—转速曲线裁剪；articulation 暴露给奖励的 effort limit 须为真实值
- 从 ANYmal 移植的力/扭矩阈值与权重须按 Go2 量级一起缩放
- 站立命令比例只采样零速度，不自动约束站立姿态
- 严格 ANYmal 奖励若收敛到三足蹲伏，须记录形态适配项（如默认关节姿态、腾空时间方差）
- torso 终止与非期望接触须分别定义，不能仅凭相似 link 名推断

## 六、修改前确认（强制）

完成同步、曲线、配置/论文/官方对照和无头诊断后，汇总证据再决定是否修改。不得边测边凭单指标打补丁。

向用户只提交：

1. **现象与证据**（最少可复现指标 + run/checkpoint/配置来源）
2. **根因判断**（已确认 / 最可能 / 待验证；排除了什么）
3. **根因→修改映射**（机制、文件、预期可观测结果；允许一组同根因改动）
4. **不修改项**（为何保留）

禁止：临时命令门控/仅失败地形分支；无论文或官方依据堆新奖励或乱调系数；用改地形深度、终止或统计口径掩盖任务失败；未确认物理/课程/观测/阶段前直接调奖励。

停下来列出拟修改项并询问是否实施。确认前不得改源码/配置、启停 4090 训练、commit 或 push。异议或新证据时更新映射并再问；沉默≠同意。改码批准≠开训批准。

## 七、修改后验证

1. 编译改动文件：
   ```bash
   uv run --python .venv/bin/python -m compileall -q <改动的.py路径...>
   ```
2. Isaac Sim 冒烟（1–2 envs）：
   ```bash
   uv run --python .venv/bin/python scripts/rsl_rl/play.py \
     --task AME-Go2-Play-v0 \
     --checkpoint <checkpoint.pt> \
     --num_envs 1 \
     --command_x 0.6 --command_y 0.0 --command_yaw 0.0
   ```
   或用无头诊断脚本：`--num_envs 2 --steps 100 --warmup_steps 20 --seed 42 --headless`。
3. 启动输出中确认观测维度与 active reward terms。
4. 若改了执行器/限位：`joints.csv` 中 effort limit 为目标值；旧失败检查点上原失效奖励应出现非零值。
5. 需重训则走下一节确认；不恢复不兼容检查点。固定检查点重复标准运动/站立测试。

## 八、收尾确认：记录、commit、push、4090 开训（一次问清）

验证完成后向用户报告：改了什么、验证结果、不确定性、是否需要重训。然后**一次询问**（分项授权，互不自动连带）：

1. 是否按本节更新 [docs/go2-policy-debug-log.md](../../../docs/go2-policy-debug-log.md)？（确认前「下一步」只写拟议+待确认）
2. 是否 `git commit`？
3. 是否 `git push`？
4. 是否在 4090 的 `ame:0` 停训并 `bash run_train.sh` 开训？（先确认代码已在 4090；开训后回报新 run）

若结论是“需要重训”，收尾提问必须包含第 4 项，不得只问记录、commit 或 push。用户同时批准 push 和开训时，默认顺序为 commit → push → 4090 `git pull --ff-only` → 停止 `ame:0.0` 旧进程 → `bash run_train.sh`。

### Debug log 写法

- 分析 run 后，补全这个 run 已有章节的**结果、判断、修改**。
- 修改验证后，把代码和这个章节放在同一个 commit 里，然后 push、开新训练。
- 新训练启动后，用实际 run 日期创建新章节，只写训练配置和“训练中”。
- 新章节先不 commit；下次分析这个 run 时再补全，并和下一次修改一起 commit。不要为新 run 单独提交日志。

### 4090 tmux

- 只读（`capture-pane` / `pgrep`）无需确认；`send-keys`、停训、开训必须获批
- 只动 `ame:0`（目录 `/home/root12/code/AME_Locomotion`），不动其他 window
- 获批后默认：对 `ame:0.0` 发一次 `C-c`，确认退出后再 `bash run_train.sh`

### Git

- 获批前不得 commit/push；只批 commit 不 push；push 前先完成并报告 commit
- 禁止顺带提交无关 dirty 文件；不 force push main/master
