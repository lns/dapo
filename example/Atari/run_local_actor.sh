#!/bin/bash

ip=localhost
port=10100

mkdir -p log
python3 -m darl.run_actor \
  --mode 'test' \
  --rnn \
  --game BreakoutNoFrameskip-v4 \
  --sub_ep "" \
  --req_ep "" \
  --push_ep "" \
  --load_path "/home/king/darl/example/Atari/out/checkpoints/iter00000001.frozen.gz" \
  --print_info

