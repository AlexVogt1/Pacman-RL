# Pacman-RL
The purpose of this repo is to conduct RL on Pacman, and create different play-style
behaviours. See [Pacman-Unity_AiPerCog](https://github.com/PipaFlores/Pacman-Unity_AiPerCog) for more info
## Structure
The `pacman_builds` folder contains different pacman executables used for training and are distiguished by the observation
space used. At the moment only `small_obs` is present, and the observation is just Pacmans loacation and movement 
direction.
## Installation & Setup
Create a conda environment using the following command
```
conda create --name pacman python=3.10.12
```
From the repo file location run the following command
```
git clone --branch release_23 https://github.com/Unity-Technologies/ml-agents.git
```
If you want to train using GPU, your will need to install Pytorch before installing `mlagents`. To install m
Pytorch with GPU (windows). Activate the `pacman conda` environment and run the following:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
To install `mlagents` Python package, activate the conda environment and run the following in the command line
```
cd /path/to/ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents
```
At this point `pacman.py` can be run to check installation of `ml-agents`is working.

To install `stable-baselines3` run the following command:
```
pip install stable-baselines3
```
If you run into a shimmy error when trying to run `train_pacman.py`, simply run 
```
pip install shimmy
```

