#!/bin/bash

mkdir -p out
# For simplicity, we write the parameters in this file
python3 -m darl.run_learner \
  --game Hopper-v2 \
  --suite robotics \
  --port 10100 \
  --policy_loss ppo \
  --value_loss vanilla \
  --base_lr 3e-4 \
  --pub_interval 1000 \
  --gpu_idx 2 \
  --nornn \
  --rollout_len 1 \

