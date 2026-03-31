# !/bin/bash
python scripts/rsl_rl/play.py \
--task AME-G1-29DOF-Play-v0 \
--checkpoint /home/fusc/workspace/LabFrame/ame_locomotion/pretrained/ame1.pt \
--num_envs 1 \
--video \
--video_length 300 \
--save_attention_weights \
--vis_attention \
