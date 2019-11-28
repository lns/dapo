#!/bin/bash

ip=localhost
port=10100

mkdir -p log
python3 -m darl.run_actor \
  --game ImmortalZealotNoReset \
  --suite arena \
  --sub_ep "tcp://$ip:$((port+0))" \
  --req_ep "tcp://$ip:$((port+1))" \
  --push_ep "tcp://$ip:$((port+2))" \
  #--load_path "out/checkpoints/iter00000001.frozen" \

