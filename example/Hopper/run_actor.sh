#!/bin/bash

ip=localhost
port=10100

mkdir -p log
python3 -m darl.run_actor \
  --game Hopper-v2 \
  --suite robotics \
  --sub_ep "tcp://$ip:$((port+0))" \
  --req_ep "tcp://$ip:$((port+1))" \
  --push_ep "tcp://$ip:$((port+2))" \

