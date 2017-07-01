
# Getting Started

以下以Ubuntu 16.04 LTS環境為準，安裝Anaconda時請一路Enter與Yes到底。

```
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash Anaconda3-4.4.0-Linux-x86_64.sh
source .bashrc
conda install pytorch torchvision -c soumith
conda install opencv
conda install libgcc
pip install gym[Atari]
sudo apt-get update
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo apt-get install fceux
pip install git+https://github.com/ppaquette/gym-super-mario/
```
