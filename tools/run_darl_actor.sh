#!/bin/bash

n=$1
game=$2
suite=$3
ip=$4
port=$5
taskid=$6

# Kill
lsof *_$taskid.log   | awk -F" " '{ print $2 }' | xargs kill
lsof *_$taskid.*.log | awk -F" " '{ print $2 }' | xargs kill
#
cd /data/rodata
source ./env.bashrc
rm core.*
cd actors
rm *.log
rm *.err
# Start pub/sub proxy
python3 -m darl.run_proxy \
  --proxy_type "PubSub" \
  --front_ep "tcp://$ip:$((port))" \
  --back_ep "ipc:///tmp/darl_pubsub_$taskid" \
  1> pub_proxy_$taskid.log 2> pub_proxy_$taskid.err &
# Start req/rep proxy
python3 -m darl.run_proxy \
  --proxy_type "ReqRep" \
  --front_ep "ipc:///tmp/darl_reqrep_$taskid" \
  --back_ep "tcp://$ip:$((port+1))" \
  1> rep_proxy_$taskid.log 2> rep_proxy_$taskid.err &
# Start push/pull proxy
#python -m darl.actor.run_proxy \
#  --proxy_type "PushPull" \
#  --front_ep "ipc:///tmp/darl_pushpull" \
#  --back_ep "tcp://$ip:$((port+2))" \
#  1> push_proxy_$taskid.log 2> push_proxy_$taskid.err &
# Start actors
for i in `seq $n`; do
  python3 -m darl.run_actor \
    --game $game \
    --suite $suite \
    --sub_ep "ipc:///tmp/darl_pubsub_$taskid" \
    --req_ep "ipc:///tmp/darl_reqrep_$taskid" \
    --push_ep "tcp://$ip:$((port+2))" \
    1> actor_$taskid.$i.log 2> actor_$taskid.$i.err &
done
wait

