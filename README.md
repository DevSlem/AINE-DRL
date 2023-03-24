# AINE-DRL

A project for the DRL baseline framework. **AINE** means "Agent IN Environment".

[Implementation](#implementation) | [Experiments](#experiments) | [Setup](#setup) |

## Implementation

AINE-DRL provides below things.

* deep reinforcement learning agents
* training in gym environment (vectorized environment also supported) and inference (rendering)
* model save/load
* YAML configuration format

If you want to know how to use, see details in [sample codes](samples/) and [Wiki](https://github.com/DevSlem/AINE-DRL/wiki).

### Agent

AINE-DRL provides basic deep reinforcement learning (DRL) agents. If you want to use them, it's helpful to read documentations in [Wiki](https://github.com/DevSlem/AINE-DRL/wiki). 

|Agent|Source Code|
|:---:|:---:|
|[REINFORCE](https://github.com/DevSlem/AINE-DRL/wiki/REINFORCE)|[reinforce](aine_drl/agent/reinforce/)|
|[A2C](https://github.com/DevSlem/AINE-DRL/wiki/A2C)|[a2c](aine_drl/agent/a2c/)|
|[PPO](https://github.com/DevSlem/AINE-DRL/wiki/PPO)|[ppo](aine_drl/agent/ppo)|
|[Recurrent PPO](https://github.com/DevSlem/AINE-DRL/wiki/Recurrent-PPO)|[ppo](aine_drl/agent/ppo/)|
|[Recurrent PPO RND](https://github.com/DevSlem/AINE-DRL/wiki/Recurrent-PPO-RND)|[ppo](aine_drl/agent/ppo/)|
|[Double DQN](https://github.com/DevSlem/AINE-DRL/wiki/Double-DQN)|[dqn](aine_drl/agent/dqn/)|


<details>
<summary><h3>TODO</h3></summary>

- [ ] Unity ML-Agents Training
- [ ] DDPG
- [ ] Prioritized Experience Replay 
- [ ] SAC
- [ ] Intrinsic Curiosity Module (ICM)
- [ ] Random Network Distillation (RND)

</details>

## Experiments

You can see our experiments (source code and result) in [experiments](experiments/). We show some recent experiments.

### BipedalWalker-v3 with PPO and SAC

Train agents in OpenAI Gym [BipedalWalker-v3](https://github.com/openai/gym/wiki/BipedalWalker-v2) which is continuous action problem.

> Note that SAC is not implemented yet.

Fig 1. BipedalWalker-v3 inference rendering (PPO):

![](images/bipedal-walker-v3-inference.webp)

* [experiment](experiments/bipedal_walker_v3/)
* [PPO configuration](config/experiments/bipedal_walker_v3_ppo.yaml)

To train the agent, enter the following command:

```bash
python experiments/bipedal_walker_v3/train.py
```

If paging file error happens, see [Paging File Error](#paging-file-error).

<!-- To inference, use:

```
python experiments/bipedal_walker_v3/train.py -m=inference
``` -->

To check graphical experiment results, use:

```bash
tensorboard --logdir=experiments/bipedal_walker_v3
```

### CartPole-v1 with No Velocity

Compare [Recurrent PPO](https://github.com/DevSlem/AINE-DRL/wiki/Recurrent-PPO) (using LSTM) and [Naive PPO](https://github.com/DevSlem/AINE-DRL/wiki/PPO) in [CartPole-v1](https://github.com/openai/gym/wiki/CartPole-v0) with No Velocity, which is [Partially Observable Markov Decision Process (POMDP)](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process) setting. Specifically, we remove **"cart velocity"** and **"pole velocity at tip"** from the observation space. This experiment shows to require memory ability in POMDP setting.

Fig 2. [CartPole-v1 with No Velocity](https://github.com/openai/gym/wiki/CartPole-v0) inference rendering (cumulative reward - Recurrent PPO: 500, Naive PPO: 28):

|Recurrent PPO|Naive PPO|
|:---:|:---:|
|![](images/cartpole-v1-with-no-velocity-inference-recurrent-ppo.webp)|![](images/cartpole-v1-with-no-velocity-inference-naive-ppo.webp)|

Fig 3. [CartPole-v1 with No Velocity](https://github.com/openai/gym/wiki/CartPole-v0) cumulative reward (purple: Recurrent PPO, sky: Naive PPO):

![](images/cartpole-v1-with-no-velocity-cumulative-reward.png)

* [experiment](experiments/cartpole_v1_no_velocity/)
* [Recurrent PPO configuration](config/experiments/cartpole_v1_no_velocity_recurrent_ppo.yaml)
* [Naive PPO configuration](config/experiments/cartpole_v1_no_velocity_ppo.yaml)

To train the agent, enter the following command:

```bash
python experiments/cartpole_v1_no_velocity/train.py
```

<!-- You can inference it using the command:

```
$ python experiments/cartpole_v1_no_velocity/train.py -m=inference
``` -->

To check graphical experiment results, use:

```bash
tensorboard --logdir=experiments/cartpole_v1_no_velocity
```

## Setup

Follow the instructions.

### Installation

Required packages:

* [Python](https://www.python.org/) 3.10.9
* [Pytorch](https://pytorch.org/) 1.11.0 - CUDA 11.3
* [Tensorboard](https://github.com/tensorflow/tensorboard) 2.12.0
* [Gym](https://github.com/openai/gym) 0.26.2
* [PyYAML](https://pyyaml.org/) 6.0

> Note that it's recommended to match the package versions. If not, it may cause API conflicts.

You can easily create an Anaconda environment. Input the command in your Anaconda shell:

```bash
conda env create -f <CONDA_ENV>.yaml
conda activate aine-drl
```

`<CONDA_ENV`> depends on your OS:

* Linux: `conda_env_linux`

If it doesn't work, install the packages manually by entering the command below:

```bash
conda create -n aine-drl python=3.10.8 -y
conda activate aine-drl
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install tensorboard==2.12.0
pip install PyYAML==6.0
pip install gym==0.26.2
pip install 'gym[all]'
pip install mlagents==0.30.0
pip install 'protobuf==3.20.*'
```

### Run

Run a sample script in [samples](samples/) directory. Enter the following command in your shell:

```bash
python samples/<FILE_NAME>
```

Example:

```bash
python samples/cartpole_v1_ppo.py
```

Then, you can see the training information in the shell:

```
[AINE-DRL] 'CartPole-v1_PPO' training start!
[AINE-DRL] training time: 1.0, global time step: 1002, cumulative reward: 16.6
[AINE-DRL] training time: 2.0, global time step: 2001, cumulative reward: 38.3
[AINE-DRL] training time: 3.1, global time step: 3000, cumulative reward: 45.8
```

The graphical result file (Tensorboard) is generated in `results` directory. You can interrupt training by `ctrl + c` and you can also retrain at the interrupted time step.

If you want to see the graphical result, input the command:

```bash
tensorboard --logdir=results
```

or

```bash
tensorboard --logdir=results/<sub_directory>
```

then, you can open the TensorBoard like below it.

Fig 4. [CartPole-v1](https://github.com/openai/gym/wiki/CartPole-v0) with PPO:

![](images/cartpole-v1-ppo-cumulative-reward-graph.png) 

* [configuration](config/samples/cartpole_v1_ppo.yaml)
* [sample code](samples/cartpole_v1_ppo.py)

### Paging File Error

When you use too many workers (e.g., greater than 8), because of too many multi parallel environments in multi threads, **"The paging file is too small for this operation to complete."** error may happen. If it happens, you can mitigate it using the command (Windows):

```bash
pip install pefile
python fixNvPe.py --input=C:\<Anaconda3 Path>\envs\aine-drl\Lib\site-packages\torch\lib\*.dll
```

`<Anaconda3 Path>` is one in which your Anaconda3 is installed.

Reference: [cobryan05/fixNvPe.py (Github)](https://gist.github.com/cobryan05/7d1fe28dd370e110a372c4d268dcb2e5)  
