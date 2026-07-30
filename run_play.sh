# !/bin/bash
python scripts/rsl_rl/play.py \
--task AME-Go2-Play-v0 \
--checkpoint logs/rsl_rl/go2_ame/2026-07-29_18-00-54/model_14000.pt \
--num_envs 1 \
--command_x 0.6 \
--command_y 0.0 \
--command_yaw 0.0
# --video \
# --video_length 300 \
# --save_attention_weights \
# --vis_attention \
