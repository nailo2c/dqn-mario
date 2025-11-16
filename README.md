# DQN

[English](README.md) | [中文](README_zh.md)

An implementation of the DQN algorithm in PyTorch, trained on Super Mario Bros and Atari Pong. The overall architecture is inspired by openai/baselines.

Warning: Training DQN requires sufficient memory. With the default replay buffer size of 1,000,000, it will use about 8 GB of RAM.

## Dependencies

- Python 3.6
- Anaconda
- PyTorch
- `gym`
- `gym[atari]`
- `ppaquette_gym_super_mario`
- `fceux`

## Getting Started

The following steps assume Ubuntu 16.04 LTS. During Anaconda installation, press Enter and Yes to accept the defaults.

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

## How to Run

- Super Mario Bros

```
xvfb-run -s "-screen 0 1400x900x24" python train_mario.py
```

- Atari Pong

```
python train_pong.py
```

## Results

- Super Mario Bros

Trained on GCP for 16 hours using 8 CPUs. 24 GB of RAM is more than sufficient, but training is hard to converge and it does not consistently clear levels. Training videos are saved by default to `/video/mario/`.

![](img/mario-dqn-16hr.gif)

- Atari Pong

Trained for 8 hours on GCP with 1 GPU (Nvidia Tesla K80) and 4 CPUs. It can reliably and decisively beat the computer. Training videos are saved by default to `/video/gym-reslults/`.

![](img/pong-dqn-8hr.gif)

## References

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [openai/baselines](https://github.com/openai/baselines)
- [transedward/pytorch-dqn](https://github.com/transedward/pytorch-dqn)
- [openai/gym](https://github.com/openai/gym)
- [ppaquette/gym-super-mario](https://github.com/ppaquette/gym-super-mario)

