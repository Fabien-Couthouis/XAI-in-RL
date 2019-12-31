# XAI-in-RL: Google Football env

## Packages installation
```
sudo apt-get update && 
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip \
cmake libopenmpi-dev python3-dev zlib1g-dev
```
## Environment installation (Anaconda)
*Note: Install Anaconda first. See [official website](https://docs.anaconda.com/anaconda/install/linux/) for further information.*
```
conda env create -f conda_env_football.yml
```

## Google football environment installation
```
git clone https://github.com/google-research/football.git
cd football
pip3 install .
```