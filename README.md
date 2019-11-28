Distributed Accelerated Reinforcement Learning
==============================================

This is an implementation of distributed reinforcement learning, used in several published work including
[Divergence-Augmented Policy Optimization](https://papers.nips.cc/paper/8842-divergence-augmented-policy-optimization)
and
[Exponentially Weighted Imitation Learning for Batched Historical Data](https://papers.nips.cc/paper/7866-exponentially-weighted-imitation-learning-for-batched-historical-data)

The project depends on a custom distributed replay memory called [memoire](https://github.com/lns/memoire). We remove the commit logs to protect sensitive IP and password information.

Examples for how to use this project for (distributed) reinforcement learning can be found in `example`.

For replicating the results of our paper, please refer to the scripts in `tools`. The main entry point is `tools/gen_atari_env.py` which can generate the shell script for running experiments in parallel, and plotting results with R. 
