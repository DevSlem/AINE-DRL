---
sort: 2
---

# Tutorial

This chapter is a tutorial for AINE-DRL. It will guide you to train a REINFORCE agent in a OpenAI Gym [CartPole-v1](https://github.com/openai/gym/wiki/CartPole-v0) environment.

## CartPole-v1

OpenAI Gym [CartPole-v1](https://github.com/openai/gym/wiki/CartPole-v0) is a classic control problem. The agent is a pole attached by an un-actuated joint to a cart, which moves along a frictionless track. The agent must learn to move the cart to keep the pole from falling over.

## Configuration

First, make a configuration file `config/cartpole_v1_reinforce.yaml`. The configuration file is a YAML file. The following is a configuration file for REINFORCE agent:

```yaml
CartPole-v1_REINFORCE:
  Env:
    type: Gym
    Config:
      id: CartPole-v1
  Train:
    Config:
      time_steps: 20000
      summary_freq: 1000
  Agent:
    gamma: 0.99
```

`CartPole-v1_REINFORCE` is the name of the configuration.

The configuration file has three sections: `Env`, `Train`, and `Agent`. The `Env` section is for environment configuration. The values of `Env/Config` key are the general arguments of `gym.make()` or `gym.vector.make()` functions. The `Train` section is for training configuration. The `Agent` section is for agent configuration `REINFORCE` agent has two settings. `gamma` and `entropy_coef`. If the setting has default value, you can skip it. See configuration details in [Agent](https://devslem.github.io/AINE-DRL/agent) docs.

> You can see entire configuration settings in `config/samples/cartpole_v1_reinforce.yaml`.

## Training Script

Next, make a training script `train.py`.

### Neural Network

AINE-DRL is based on [PyTorch](https://pytorch.org/) 1.11.0 - CUDA 11.3 library.

To make a REINFORCE network, you need to implement `REINFORCENetwork` interface class. The following is a training script:

```python
import torch.nn as nn
import torch.optim as optim

import aine_drl
import aine_drl.agent as agent
from aine_drl.factory import AgentFactory, AINEInferenceFactory, AINETrainFactory
from aine_drl.policy import CategoricalPolicy


class CartPoleREINFORCENet(nn.Module, agent.REINFORCENetwork):    
    def __init__(self, obs_features, num_actions) -> None:
        super().__init__()
        
        # policy layer
        self.policy_net = nn.Sequential(
            nn.Linear(obs_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            CategoricalPolicy(64, num_actions)
        )
        
    def model(self) -> nn.Module:
        return self.policy_net
    
    def forward(self, obs: aine_drl.Observation) -> aine_drl.PolicyDist:
        return self.policy_net(obs.items[0])
```

`Observation` has a tuple of observation `Tensor` in `items` field. Environment may provide multiple observations (e.g., image and vector). In this case, CartPole-v1 provides only vector observation. So, `items[0]` is the vector observation.

`CategoricalPolicy` is a linear layer whose output is a categorical policy distribution.

### Agent Factory

To make a REINFORCE agent, you need to implement `AgentFactory` interface class. The following is a training script:

```python
class REINFORCEFactory(AgentFactory):
    def make(self, env: aine_drl.Env, config_dict: dict) -> agent.Agent:
        config = agent.REINFORCEConfig(**config_dict)
        
        network = CartPoleREINFORCENet(
            obs_features=env.obs_spaces[0][0],
            num_actions=env.action_space.discrete[0]
        )
        
        trainer = aine_drl.Trainer(optim.Adam(
            network.parameters(),
            lr=0.001
        )).enable_grad_clip(network.parameters(), max_norm=5.0)
        
        return agent.REINFORCE(
            config,
            network,
            trainer,
        )
```

`config_dict` is a dictionary of `Agent` section in the configuration file. 

`aine_drl.Env` is a AINE-DRL environment wrapper class. `env.obs_spaces` is a tuple of `ObservationSpace` instances. `ObservationSpace` class represents the shape of observation space. For example, vector space is `(4,)` and image space is `(600, 400, 3)`. `env.action_space` is an action space. `env.action_space.discrete` is a tuple whose each element is the number of actions of each action branch. `env.action_space.continuous` is the number of continuous actions. `env.action_space.discrete[0]` is the number of discrete actions of the first discrete action branch.

`Trainer` is a optimizer wrapper class. `Trainer` has `enable_grad_clip()` method to enable gradient clipping. It can prevent gradient explosion which causes training instability.

### Main Code

Now write a main code to start training:

```python
if __name__ == "__main__": 
    config_path = "config/cartpole_v1_reinforce.yaml"
    
    AINETrainFactory.from_yaml(config_path) \
        .make_env() \
        .make_agent(REINFORCEFactory()) \
        .ready() \
        .train() \
        .close()
```

`AINETrainFactory` is a factory class to make a `Train` instance. You can use methods by chain rule! `AINETrainFactory.from_yaml()` method loads a configuration file. `AINETrainFactory.make_env()` method makes a training environment. In this case OpenAI Gym CartPole-v1. `AINETrainFactory.make_agent()` method makes an agent (i.e., REINFORCE agent). `AINETrainFactory.ready()` method makes a `Train` instance to prepare the training. `Train.train()` method starts training. `Train.close()` method closes the training safely.

> You can see entire training codes in `samples/cartpole_v1_reinforce.py`.

## Start Training!

If you follow the above steps, run the training script by entering the following command:

```bash
python train.py
```

then you can see the training information in your shell:

```
+----------------------------------------------------+
| AINE-DRL Training Start!                           |
|====================================================|
| ID: CartPole-v1_REINFORCE                          |
| Output Path: results/CartPole-v1_REINFORCE         |
|----------------------------------------------------|
| Training INFO:                                     |
|     number of environments: 1                      |
|     total time steps: 20000                        |
|     summary frequency: 1000                        |
|     agent save frequency: 10000                    |
|----------------------------------------------------|
| Agent INFO:                                        |
|     name: REINFORCE                                |
|     device: cpu                                    |
+----------------------------------------------------+

[AINE-DRL] training time: 0.43, time steps: 1000, cumulated reward: 22.23
[AINE-DRL] training time: 0.91, time steps: 2000, cumulated reward: 33.30
[AINE-DRL] training time: 1.35, time steps: 3000, cumulated reward: 48.10
```

When the training is finished, you can see the training result files (tensorboard, log message, agent save file) in the `results/CartPole-v1_REINFORCE` directory.

If you want to see the graphical training result, use:

```bash
tensorboard --logdir=results
```

## Inference

Now, let's inference the trained agent!

First, add `CartPole-v1_REINFORCE/Inference` section in the configuration file `config/cartpole_v1_reinforce.yaml`:

```yaml
Inference:
  Config:
    episodes: 3
```

`episodes` is the number of episodes to inference.

Then, add an inference code in `train.py`:

```python
AINEInferenceFactory.from_yaml(config_path) \
    .make_env() \
    .make_agent(REINFORCEFactory()) \
    .ready() \
    .inference() \
    .close()
```

It's similar to the `AINETrainFactory` code. `AINEInferenceFactory.ready()` method makes an `Inference` instance. `Inference.inference()` method starts inference. `Inference.close()` method closes the inference safely.

Now, run the training script by entering the following command:

```bash
python train.py
```

then you can see the inference information in your shell:

```
[AINE-DRL] Training is already finished.
[AINE-DRL] Agent is successfully loaded from: results/CartPole-v1_REINFORCE/agent.pt
[AINE-DRL] inference - episode: 0, cumulative reward: 302.00
[AINE-DRL] inference - episode: 1, cumulative reward: 295.00
[AINE-DRL] inference - episode: 2, cumulative reward: 328.00
[AINE-DRL] Inference is finished.
```

You can see the real-time rendering where the cart moves!

> You may not satisfy the inference result, because cumulative reward is not high. It is because the training is not enough. You can increase the training time steps by changing the `Train/Config/time_steps` setting in the configuration file.

AINE-DRL provides different inference options. You can export the inference result as a GIF file or pictures (video is not currently supported). Change the `Inference/Config` section in the configuration file:

```yaml
Inference:
  Config:
    episodes: 3
    export: gif # default: render_only
```

then you can see the GIF files in the `results/CartPole-v1_REINFORCE/exports/gifs` directory.

