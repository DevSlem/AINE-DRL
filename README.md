# AINE DRL

A project for the DRL framework. AINE is the team name which means "Agent IN Environment".

## Implemented Algorithm

- [x] [DQN](aine_drl/agent/dqn.py)
- [x] [Double DQN](aine_drl/agent/dqn.py)
- [x] [REINFORCE](aine_drl/agent/reinforce.py)

## TODO

- [ ] SARSA
- [ ] A2C
- [ ] PPO

## Development Environment

* Python 3.7.13
* Pytorch 1.11.0 - CUDA 11.3

## Run

If you run a test file in `test` directory, you can see the result in `results` directory. Input below commands.

```
tensorboard --logdir=results
```

or

```
tensorboard --logdir=results/<sub_directory>
```

## Convention

You must write specific comments in English.  
One script file per one class.  

### Naming Convention

Script file name is `snake_case`.  
Class name is `UpperCamelCase`.  
Method name is `snake_case`.  
Variable name is `snake_case`.

The name of the agent which inherits `Agent` class must be `<Name>Agent`.