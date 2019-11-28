#!/bin/bash

mkdir -p out
# For simplicity, we write the parameters in this file
python3 -m darl.run_learner \
  --game "BreakoutNoFrameskip-v4" \
  --suite "atari" \
  --port 10100 \
  --policy_loss ppo \
  --value_loss vtrace \
  --pub_interval 1000 \
  --devices 0,1,2,3,6,7 \
  --rollout_len 32 \
  --batch_size 4096 \
  --rnn \

