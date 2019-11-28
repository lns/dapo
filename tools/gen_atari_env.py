#!/usr/bin/env python
# -*- coding:utf-8 -*-

GPU_SERVER_IP='8.8.8.8' # IP of local GPU server

game_list = ['AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', \
    'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', \
    'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival', \
    'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk', \
    'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', \
    'Gravitar', 'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', \
    'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', \
    'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan', \
    'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', \
    'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner', \
    'Tennis', 'TimePilot', 'Tutankham', 'Venture', 'VideoPinball', \
    'WizardOfWor', 'YarsRevenge', 'Zaxxon']

#robot_list = ['HalfCheetah','Hopper','InvertedDoublePendulum','InvertedPendulum', 'Reacher','Swimmer','Walker2d']
robot_list = ['HalfCheetah','Hopper','InvertedPendulum', 'Reacher','Swimmer','Walker2d'] # lr3e-4, lam0.95
hard_list = ['Frostbite','Freeway','MontezumaRevenge','Venture']

main_list = ['BeamRider', 'Breakout', 'Qbert', 'Seaquest', 'SpaceInvaders'] # Used as example in DPER
exp_list = hard_list
#for each in main_list:
#    exp_list.remove(each)

alg_list = ['mkl_on_0.5ns_c0.9','mkl_on_0.5ns_c0.5']

algs = { \
    'mkl_off'      : 'pgis vanilla KL 0.0 1e-3 0.0 off 1.0 0.0 0.0 0.0 0.0 0.9', \
    'mkl_on'       : 'pgis vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.0 0.0 0.0 0.9', \
    'ppo_off'      : 'ppo  vanilla KL 0.0 1e-3 0.0 off 1.0 0.0 0.0 0.0 0.0 0.9', \
    'ppo_on'       : 'ppo  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.0 0.0 0.0 0.9', \
    'mkl_off_1sD'  : 'pgis vanilla KL 0.0 1e-3 0.0 off 1.0 0.0 1.0 0.0 0.0 0.9', \
    'mkl_off_nsD'  : 'pgis vanilla KL 0.0 1e-3 0.0 off 1.0 0.0 1.0 0.0 1.0 0.9', \
    'ppo_off_1sD'  : 'ppo  vanilla KL 0.0 1e-3 0.0 off 1.0 0.0 1.0 0.0 0.0 0.9', \
    'ppo_off_nsD'  : 'ppo  vanilla KL 0.0 1e-3 0.0 off 1.0 0.0 1.0 0.0 1.0 0.9', \
    'mkl_on_1sD'   : 'pgis vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'mkl_on_nsD'   : 'pgis vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'mkl_on_0.5ns_c0.9'   : 'pgis vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.5 0.0 0.9 0.9', \
    'mkl_on_0.5ns_c0.5'   : 'pgis vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.5 0.0 0.5 0.9', \
    'mkl_on_0.1ns_c0.9'   : 'pgis vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.1 0.0 0.9 0.9', \
    'acer_on'      : 'acer vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.0 0.0 0.0 0.9', \
    'acer_on_1sD'  : 'acer vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'acer_on_nsD'  : 'acer vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'acerinf_on_1sD'  : 'acer vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'acerinf_on_nsD'  : 'acer vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'ppo_on_1sD'   : 'ppo  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'ppo_on_nsD'   : 'ppo  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'inf_on_1sD'   : 'pgis vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'inf_on_nsD'   : 'pgis vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'mtv_on_1sD'   : 'pgis vtrace  TV 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'mtv_on_nsD'   : 'pgis vtrace  TV 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'ppg_on'       : 'ppg  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.0 0.0 0.0 0.9', \
    'pkl_on_1sD'   : 'ppg  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'pkl_on_nsD'   : 'ppg  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'pkl_on_nsD_c0.5'   : 'ppg  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.5 0.9', \
    'pkl_on_0.5nsD'   : 'ppg  vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.5 0.0 1.0 0.9', \
    'akl_on_1sD'   : 'acerg vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'akl_on_nsD'   : 'acerg vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'aif_on_1sD'   : 'acerg vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'aif_on_nsD'   : 'acerg vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'pif_on_1sD'   : 'ppg  vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'pif_on_nsD'   : 'ppg  vtrace INF 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'mkl_on_1sD_pgis2'  : 'pgis2 vtrace KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'mkl_on_nsD_pgis2'  : 'pgis2 vtrace KL 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'mkl2_on_1sD'  : 'pgis2 vtrace KL2 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 0.0 0.9', \
    'mkl2_on_nsD'  : 'pgis2 vtrace KL2 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'mkl2_on_nsD_r2'  : 'pgis2 vtrace KL2 0.0 1e-3 0.0 on  1.0 0.0 0.1 0.0 1.0 0.9', \
    'mkl2_on_nsD_r3'  : 'pgis2 vtrace KL2 0.0 1e-3 0.0 on  1.0 0.0 0.5 0.0 1.0 0.9', \
    'mkl2_on_nsD_sq'  : 'pgis2 vtrace KL2 0.0 1e-3 0.0 on  1.0 0.0 1.0 0.0 1.0 0.9', \
    'marwil_on'       : 'marwil vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.0 0.0 0.0 0.9', \
    'vrmarwil_on'     : 'vrmarwil vtrace  KL 0.0 1e-3 0.0 on  1.0 0.0 0.0 0.0 0.0 0.9', \
    }

def gen_sh(norun=False):
  n_actor_per_machine = 16
  n_slots = 64
  cmds = [''] * 8
  if False:
    exp = game_list
    #for each in exp_list:
    #  exp.remove(each)
  else:
    #exp = robot_list
    exp = exp_list

  #print(len(game_list))
  gpu_id = 0
  for each in exp:
    #game = each+'-v2'
    game = each+'NoFrameskip-v4'
    for alg in alg_list:
      for test_number in range(1,2):
        port = 9420 + gpu_id * 100
        dirname = alg+'_2e8_rnn_32_0_16k_a%d_lr1e3' % (n_actor_per_machine)
        cmd = 'run_exp ~/darl/bench/%s/%s_t%d %s atari %d %d %d %d %s 100 2e8 rnn 32 0 16384' % \
            (each, dirname, test_number, game, n_actor_per_machine, n_slots, \
             port, gpu_id, algs[alg])
        cmds[gpu_id] += cmd + '\n'
        gpu_id = (gpu_id + 1) % len(cmds)

  for i in range(len(cmds)):
    with open("run%d.sh" % i, "w") as f:
      f.write('''
#/bin/bash

serverlist=servers/bx_n1_%02d
local_ip=%s
sleep_time=%d

source ../tools/auto_func.sh

''' % (i, GPU_SERVER_IP, 7200))
      f.write(cmds[i])
  if not norun:
    from os import system
    for i in range(len(cmds)):
      system("nohup sh +x ./run%d.sh > logs/run%d.log 2>&1 &" % (i, i))

def gen_plot():
  from os import system
  for each in exp_list:
    system("sh -c 'cd %s; R -q -f ../make_plot.R; cd ..' &" % each)

def gen_text():
  for each in game_list:
    print('<img align="left" src="./figs/%s.png">' % (each))

def destroy():
  from os import system
  for each in game_list:
    system("sh -c 'mv ./%s obsolated/' &" % each)


if __name__ == '__main__':
  import sys
  if len(sys.argv) <= 1:
    print(' Usage: %s [PLOT|EXP|NORUN]' % sys.argv[0])
    exit(0)
  if sys.argv[1].lower() == 'plot':
    gen_plot()
  elif sys.argv[1].lower() == 'exp':
    gen_sh(norun=False)
  elif sys.argv[1].lower() == 'norun':
    gen_sh(norun=True)
  elif sys.argv[1].lower() == 'destroy':
    destroy()
  elif sys.argv[1].lower() == 'copy':
    copy()
  else:
    raise NotImplementedError

