# Pacman 
## Overview

## Structure

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
Pytorch with GPU (windows). Activate the `pacman conda` environment and run the follwoing:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
To intsall `mlagents` Python package, activate the conda environment and run the following in the command line
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

