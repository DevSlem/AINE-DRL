# AINE DRL

A project for the DRL framework. AINE is the team name which means "Agent IN Environment".

## Implemented Algorithm

- [x] [DQN](aine_drl/agent/dqn.py)
- [x] [Double DQN](aine_drl/agent/dqn.py)
- [x] [REINFORCE](aine_drl/agent/reinforce.py)
- [x] [A2C](aine_drl/agent/a2c.py)

## TODO

- [ ] SARSA
- [ ] PPO

## Development Environment

* Python 3.7.13
* Pytorch 1.11.0 - CUDA 11.3
* Tensorboard 2.10.0
* Gym 0.25.2

You can easily create an Anaconda environment by using the command:

```
conda env create -f aine_drl_env.yaml
```

## Run

If you run a sample code file in [samples](samples/) directory, you can see the result in `results` directory. Input below commands.

```
tensorboard --logdir=results
```

or

```
tensorboard --logdir=results/<sub_directory>
```

then, you can see a tensorboard like below it.

![](images/cartpole-v1-reinforce-cumulative-reward-graph.png)

## Convention

You must write specific comments in English.  
One script file per one class.  

### Naming Convention

Script file name is `snake_case`.  
Class name is `UpperCamelCase`.  
Method name is `snake_case`.  
Variable name is `snake_case`.
