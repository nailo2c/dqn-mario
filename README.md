[English](#english) | [中文](#中文)

<h1 id="english">DQN (English)</h1>

Implemented the DQN algorithm using PyTorch to train on Super Mario Bros. and Atari Pong. The overall architecture is based on openai/baselines.

*Warning*: Please ensure you have sufficient memory when training DQN. For example, with the default Replay Buffer size of 1,000,000, it will consume at least 8GB of memory.

# Dependencies

*   Python 3.6
*   Anaconda
*   PyTorch
*   gym
*   gym[atari]
*   ppaquette_gym_super_mario
*   fceux

# Getting Started

The following instructions are based on an Ubuntu 16.04 LTS environment. When installing Anaconda, please accept the defaults by pressing Enter and typing "Yes".

```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x88_64.sh
bash Anaconda3-4.4.0-Linux-x86_64.sh
source .bashrc
conda install pytorch torchvision -c soumith
conda install libgcc
pip install gym[Atari]
sudo apt-get update
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo apt-get install fceux
pip install git+https://github.com/ppaquette/gym-super-mario/
```

# How to run

*   super-mario-bros
```
xvfb-run -s "-screen 0 1400x900x24" python train_mario.py
```

*   atari-pong
```
python train_pong.py
```

# Result

*   Super-Mario-Bros

Trained on GCP with 8 CPUs for 16 hours. 24GB of RAM was more than sufficient, but the model had difficulty converging and could not consistently pass the level.
The training videos are saved by default in `/video/mario/`.

![](img/mario-dqn-16hr.gif)

*   Atari-Pong

Trained on GCP with 1 GPU (Nvidia Tesla K80) and 4 CPUs for 8 hours. The model can consistently and significantly outperform the computer opponent.
The training videos are saved by default in `/video/gym-reslults/`.

![](img/pong-dqn-8hr.gif)

# References

[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
[openai/baselines](https://github.com/openai/baselines)
[transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)
[openai/gym](https://github.com/openai/gym)
[ppaquette/gym-super-mario](https://github.com/ppaquette/gym-super-mario)

---

<h1 id="中文">DQN (中文)</h1>

使用PyTorch實作DQN演算法，並訓練super-mario-bros以及atari-pong，整體架構參考openai/baselines。  

*Warning*：訓練DQN請開足夠的記憶體，Replay Buffer以預設值1000000為例至少會使用約8G的記憶體。
  
# Dependencies

* Python 3.6
* Anaconda
* PyTorch
* gym
* gym[atari]
* ppaquette_gym_super_mario
* fceux
  
# Getting Started

以下以Ubuntu 16.04 LTS環境為準，安裝Anaconda時請一路Enter與Yes到底。

```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash Anaconda3-4.4.0-Linux-x86_64.sh
source .bashrc
conda install pytorch torchvision -c soumith
conda install libgcc
pip install gym[Atari]
sudo apt-get update
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo apt-get install fceux
pip install git+https://github.com/ppaquette/gym-super-mario/
```
  
# How to run

* super-mario-bros
```
xvfb-run -s "-screen 0 1400x900x24" python train_mario.py
```
  
* atari-pong
```
python train_pong.py
```

# Result

* Super-Mario-Bros

使用8顆cpu在GCP上跑16個小時，RAM開24G非常足夠，但很難收斂，無法穩定過關。  
訓練的影像預設位置在/video/mario/。

![](img/mario-dqn-16hr.gif)

* Atari-Pong

使用1張GPU(Nvidia Tesla K80)加4顆cpu在GCP上跑8個小時，能夠穩定大幅贏電腦。  
訓練的影像預設位置在/video/gym-reslults/。

![](img/pong-dqn-8hr.gif)


# References

[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[openai/baselines](https://github.com/openai/baselines)  
[transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)  
[openai/gym](https://github.com/openai/gym)  
[ppaquette/gym-super-mario](https://github.com/ppaquette/gym-super-mario)  
