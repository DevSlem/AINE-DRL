# Installation

First, you need to install [Python](https://www.python.org/) 3.9 or higher but we recommend to install Python 3.9 version. 

If you use Anaconda, create an Anaconda environment first by entering the command below (optional):

```bash
conda create -n aine-drl python=3.9 -y
conda activate aine-drl
```

You may prefer to use NVIDIA CUDA due to the training speed. In this case, you need to install manually PyTorch with CUDA (this step can be skipped if cpu is only used):

```bash
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Now, install AINE-DRL package by entering the command below:

```
pip install aine-drl
```

Required packages:

* [Python](https://www.python.org/) 3.9 or higher
* [Pytorch](https://pytorch.org/) 1.11.0 - CUDA 11.3
* [Tensorboard](https://github.com/tensorflow/tensorboard) 2.12.0
* [PyYAML](https://pyyaml.org/) 6.0
* [Gym](https://github.com/openai/gym) 0.26.2
* [ML-Agents](https://github.com/Unity-Technologies/ml-agents/tree/release_20) 0.30.0
* [Protocol Buffer](https://protobuf.dev/getting-started/pythontutorial/) 3.20

## Manual Installation

If you fail to install from PyPi, you should clone the repository and install the required packages manually. First, clone the repository:

```bash
git clone https://github.com/DevSlem/AINE-DRL.git
```

Then, install the required packages:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard==2.12.0
pip install PyYAML==6.0
pip install gym==0.26.2
pip install gym[all]
pip install mlagents==0.30.0
pip install protobuf==3.20.*
```

Depending on the shell you are using (e.g., [Zsh](https://www.zsh.org/)), you may need to add quotes like `'gym[all]'` and `'protobuf==3.20.*'`.

## Local Installation

If you intend to make modifications to AINE-DRL package, you should install the package from the cloned repository rather than PyPi. From the repository's root directory, use:

```
pip install -e .
```

Another way is to install `aine_drl` module locally in your own project. From your project directory, use:

```bash
git init
git remote add -f origin https://github.com/DevSlem/AINE-DRL.git
git config core.sparseCheckout true
echo "aine_drl/" >> .git/info/sparse-checkout
git pull origin main
```

Of course, you need to install required python packages to use this module.
