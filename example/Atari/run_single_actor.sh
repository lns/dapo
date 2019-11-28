#!/bin/bash

ip=localhost
port=10100

mkdir -p log
python3 -m darl.run_actor \
  --game BreakoutNoFrameskip-v4 \
  --suite 'atari' \
  --sub_ep "tcp://$ip:$((port+0))" \
  --req_ep "tcp://$ip:$((port+1))" \
  --push_ep "tcp://$ip:$((port+2))" \

