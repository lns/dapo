#!/bin/bash

mkdir -p out
# For simplicity, we write the parameters in this file
python3 -m darl.run_learner \
  --game ImmortalZealotNoReset \
  --suite arena \
  --port 10100 \
  --policy_loss ppo \
  --value_loss vanilla \
  --base_lr 3e-4 \
  --pub_interval 1000 \
  --gpu_idx 6 \
  --rnn \
  --rollout_len 4 \

