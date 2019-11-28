#/bin/bash

export DARL_ROOT=$HOME/darl

SCRIPT_ENV=$HOME/darl/bench

function run_exp() {
  dir=$1
  envname=$2
  suite=$3
  n_actor=$4
  n_slot=$5
  local_port=$6
  gpu_idx=$7
  #
  ploss=$8
  vloss=$9
  reg=${10}
  priority=${11}
  base_lr=${12}
  final_lr=${13}
  adv_est=${14}
  adv_coef=${15}
  adv_off=${16}
  reg_coef=${17}
  ent_coef=${18}
  cbarD=${19}
  mix_lambda=${20}
  pub_interval=${21}
  total_samples=${22}
  rnn=${23}
  rollout_len=${24}
  max_episode=${25}
  max_step=${26}
  #
  if [ "$rnn" = "rnn" ]; then
    echo "RNN"
  elif [ "$rnn" = "nornn" ]; then
    echo "NO RNN"
  else
    echo "Unknown option '$rnn'"
    error
  fi
  #
  echo `date` "Running $dir on $envname"
  cd $SCRIPT_ENV
  # kill actors first to clean
  cd $SCRIPT_ENV
  lsof actors_$gpu_idx.log | awk -F" " '{ print $2 }' | xargs kill
  sleep 10
  # start
  $DARL_ROOT/tools/master_run3.sh $serverlist $envname $suite $n_actor $local_ip $local_port $gpu_idx \
    > actors_$gpu_idx.log 2>&1 &
  mkdir -p $dir/out
  cd $dir
  rm -f *.log *.err
  python3 -m darl.run_learner \
    --game $envname \
    --suite $suite \
    --$rnn \
    --rollout_len $rollout_len \
    --port $local_port \
    --policy_loss $ploss \
    --value_loss $vloss \
    --reg $reg \
    --priority_exponent $priority \
    --base_lr $base_lr \
    --final_lr $final_lr \
    --adv_est $adv_est \
    --adv_coef $adv_coef \
    --adv_off $adv_off \
    --reg_coef $reg_coef \
    --ent_coef $ent_coef \
    --cbarD $cbarD \
    --mix_lambda $mix_lambda \
    --n_slot $n_slot \
    --gpu_idx $gpu_idx \
    --pub_interval $pub_interval \
    --max_episode $max_episode \
    --max_step $max_step \
    --total_samples $total_samples \
    --pickle_protocol 3 \
    > run.log 2> run.err &
  # sleep
  sleep $sleep_time
  # kill actors first to get scores of unfinished episodes
  cd $SCRIPT_ENV
  lsof actors_$gpu_idx.log | awk -F" " '{ print $2 }' | xargs kill
  sleep 10
  # then kill the learner
  cd $dir
  lsof run.log | awk -F" " '{ print $2 }' | xargs kill
  sleep 61 # Related to cat /proc/sys/net/ipv4/tcp_fin_timeout ?
  # we occasionally encounter 'Address already in use' error when starting new learner
}

