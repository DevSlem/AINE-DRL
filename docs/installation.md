# Installation

Required packages:

* [Python](https://www.python.org/) 3.10.8
* [Pytorch](https://pytorch.org/) 1.11.0 - CUDA 11.3
* [Tensorboard](https://github.com/tensorflow/tensorboard) 2.12.0
* [PyYAML](https://pyyaml.org/) 6.0
* [Gym](https://github.com/openai/gym) 0.26.2
* [ML-Agents](https://github.com/Unity-Technologies/ml-agents/tree/release_20) 0.30.0
* [Protocol Buffer](https://protobuf.dev/getting-started/pythontutorial/) 3.20

If you use Anaconda, create an anaconda environment first by entering the command below:

```bash
conda create -n aine-drl python=3.10.8 -y
conda activate aine-drl
```

Note that **Windows** users may happen to Numpy dependency error. It's because [Numpy 1.21.2](https://numpy.org/doc/stable/release/1.21.1-notes.html) does not support Python 3.10. To solve this problem, it's highly recommended to first install Numpy 1.21.2 from the Anaconda command. (**Linux** users can skip this step):

```bash
conda install numpy==1.21.2
```

Install the packages by entering the command below:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard==2.12.0
pip install PyYAML==6.0
pip install gym==0.26.2
pip install 'gym[all]'
pip install mlagents==0.30.0
pip install 'protobuf==3.20.*'
```

The **quote** marks are based on [zsh](https://www.zsh.org/) shell. If you use another shell (e.g., [powershell](https://learn.microsoft.com/en-us/powershell/)), you may need to remove the quote marks.

> This installation guide is based on Linux and Anaconda environment. Please give me a pull request if you have a better installation guide.
