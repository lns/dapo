#!/bin/bash
if [[ -z $5 ]]; then
  echo " Usage: $0 n_actor game ip port taskid"
  exit
fi

# Number of actors (16)
n_actor=$1
# Game Name (BreakoutNoFrameskip-v4)
game=$2
# IP of learner instance
ip=$3
# Port of learner instance
port=$4
# Task ID
taskid=$5

mkdir -p log
# Start pub/sub proxy
python3 -m darl.run_proxy \
  --proxy_type "PubSub" \
  --front_ep "tcp://$ip:$((port))" \
  --back_ep "ipc:///tmp/darl_atari_pubsub_$taskid" \
  1> log/pub_proxy_$taskid.log 2> log/pub_proxy_$taskid.err &
# Start req/rep proxy
python3 -m darl.run_proxy \
  --proxy_type "ReqRep" \
  --front_ep "ipc:///tmp/darl_atari_reqrep_$taskid" \
  --back_ep "tcp://$ip:$((port+1))" \
  1> log/rep_proxy_$taskid.log 2> log/rep_proxy_$taskid.err &
# Start actors
for i in `seq $n_actor`; do
  python3 -m darl.run_actor \
    --mode 'train' \
    --game $game \
    --sub_ep "ipc:///tmp/darl_atari_pubsub_$taskid" \
    --req_ep "ipc:///tmp/darl_atari_reqrep_$taskid" \
    --push_ep "tcp://$ip:$((port+2))" \
    1> log/actor_$taskid.$i.log 2> log/actor_$taskid.$i.err &
done
wait

