#!/bin/bash
#SBATCH -p bigbatch
#SBATCH -N 1
#SBATCH -J pacman_train_behaviour
#SBATCH -o ./logs/cluster/behavlets/pacman_train_behaviour.%N.%j.out
#SBATCH -e ./logs/cluster/behavlets/pacman_train_behaviour.%N.%j.err

echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo Running on $HOSTNAME...
echo Running on $HOSTNAME... >&2

source ~/.bashrc
cd ~/Pacman-RL
conda activate pacman

python train_behaviour_pacman.py --json_path="exp/behaviour/p1abcd_003"

echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"
