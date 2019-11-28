#!/bin/bash
if [[ -z $7 ]]; then
  echo " Usage: $0 ip_list game suite n_actor learner_ip learner_port gpu_idx"
  exit
fi

ip_list=$1
game=$2
suite=$3
n_actor=$4
remote_ip=$5
remote_port=$6
task_id=$7

function main() {
  ip=$1
  name=$2
  port=$3
  pswd=$4
  echo $ip
  $DARL_ROOT/tools/scp.exp $DARL_ROOT/tools/run_darl_actor.sh /tmp $name "$pswd" $ip $port $verbose
  # 16 means starting 16 actors on each machine
  $DARL_ROOT/tools/sh.exp "sh +x /tmp/run_darl_actor.sh $n_actor $game $suite $remote_ip $remote_port $task_id" $name "$pswd" $ip $port $verbose
}

# Main loop
count=0
verbose=1
while read line; do
  #
  let count=$count+1
  let mod=$count%1024
  if [ $mod -eq 0 ]; then
    wait
    verbose=1
  fi
  sleep 0.01
  #
  main $line $verbose &
  verbose=0
done < $ip_list
wait
