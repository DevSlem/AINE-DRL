# AINE DRL

A project for the DRL framework. **AINE** is the team name which means "Agent IN Environment".

AINE-DRL supports below things.

* deep reinforcement learning agents
* training interrupt and model save
* training with gym environment (vectorized environment also supported)
* rendering gym environment with inference mode

If you want to know how to use, see details in [sample codes](samples/).

## Algorithm

### Implemented

- [x] [DQN](aine_drl/agent/dqn.py)
- [x] [Double DQN](aine_drl/agent/dqn.py)
- [x] [REINFORCE](aine_drl/agent/reinforce.py)
- [x] [A2C](aine_drl/agent/a2c.py)
- [x] [PPO](aine_drl/agent/ppo/ppo.py)

### TODO

- [ ] SARSA
- [ ] Prioritized Experience Replay 
- [ ] A3C
- [ ] SAC
- [ ] Intrinsic Curiosity Module (ICM)
## Experiment

### BipedalWalker-v3 with PPO

See details of [BipedalWalker-v3](https://github.com/openai/gym/wiki/BipedalWalker-v2) environment.

Fig 1. [BipedalWalker-v3](https://github.com/openai/gym/wiki/BipedalWalker-v2) with PPO agent:

![](images/bipedal-walker-v3-ppo-cumulative-reward-graph.png)

* gray - no gradient clipping
* sky - gradient clipping with 0.5
* pink - gradient clipping with 5.0

Source Code: [bipedal_walker_v3_ppo.py](samples/bipedal_walker_v3_ppo.py)

## Setup

Follow the instructions.

### Installation

* Python 3.7.13
* Pytorch 1.11.0 - CUDA 11.3
* Tensorboard 2.10.0
* Gym 0.25.2
* PyYAML 6.0

You can easily create an Anaconda environment. Input the command in your Anaconda shell:

```
$ conda env create -f conda_env.yaml
$ conda activate aine-drl
```

> Note that it's recommended to match the package versions. If not, it may cause API conflicts.

### Run

Run a sample script in [samples](samples/) directory. Input the command in your shell:

```
$ python samples/<file_name>
```

Example:

```
$ python samples/cartpole_v1_ppo.py
```

Then, you can see the result in the shell. The result file is also generated in `results` directory. You can interrupt training by `ctrl + c`. You can also retrain at the interrupted time step.

If you want to see the summarized results, input the command:

```
$ tensorboard --logdir=results
```

or

```
$ tensorboard --logdir=results/<sub_directory>
```

then, you can open the TensorBoard like below it.

Fig 2. [CartPole-v1](https://github.com/openai/gym/wiki/CartPole-v0) with PPO agent:

![](images/cartpole-v1-ppo-cumulative-reward-graph.png) 

* [configuration](config/cartpole_v1_ppo.yaml)
* [sample code](samples/cartpole_v1_ppo.py)

## Module

* [aine_drl](aine_drl/)
  * [agent](aine_drl/agent/)*
  * [drl_util](aine_drl/drl_util/)
  * [policy](aine_drl/policy/)*
  * [training](aine_drl/training/)
  * [trajectory](aine_drl/trajectory/)
  * [util](aine_drl/util/)
  * [experience](aine_drl/experience.py)*
  * [network](aine_drl/network.py)*

> Note that `*` indicates you can directly access API in the module by `aine_drl`.
