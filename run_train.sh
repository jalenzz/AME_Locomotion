#!/bin/bash
python scripts/rsl_rl/train.py \
--task AME-Go2-v0 \
--max_iterations 15000 \
--headless \
--num_envs 1024
# --run_name  \
# --resume \
# --load_run  \
# --checkpoint  \
