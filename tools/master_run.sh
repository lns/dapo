#!/bin/bash
if [[ -z $1 ]]; then
  echo " Usage: $0 ip_list"
  exit
fi

ip_list=$1

# Login test
login_test='echo Pass'
# Yum update
yum_update='yes | yum update'
# Docker install
docker_install='sh -c "./install.sh > install.log 2> install.err"'
# Repair yum
repair_yum='sh -c "./repair_rpm.sh > repair.log 2> repair.err"'
# Docker fix
dockerfix='mv -r /data/rodata/* .; rm -rf /data/rodata'
# Start actors
start_actors='cd /data/rodata; source env.bashrc; cd actors; python actor.py'
# Start dockers
start_dockers='cd /root; source env.bashrc; cd actors; python actor.py'

# Main loop
count=0
verbose=1
while read line; do
  #
  let count=$count+1
  let mod=$count%1280
  if [ $mod -eq 0 ]; then
    wait
    verbose=1
  fi
  sleep 0.01
  #
  array=($line)
  ip=${array[0]}
  name=${array[1]}
  port=${array[2]}
  pswd=${array[3]}
  echo $ip
  ./sh.exp "$sync_only" $name "$pswd" $ip $port $verbose &
  verbose=0
done < $ip_list
wait
