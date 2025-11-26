#!/bin/bash
#SBATCH -p bigbatch
#SBATCH -N 1
#Sbatch --nodelist=mscluster[82,85,86]
#SBATCH -J pacman_train
#SBATCH -o ./logs/cluster/pacman_train.%N.%j.out
#SBATCH -e ./logs/cluster/pacman_train.%N.%j.err
#SBATCH -x mscluster[8,9,35,42,44,48,51,54,57,59,61,62,65,66,68,71,72,75,76,106]

echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
echo Running on $HOSTNAME...
echo Running on $HOSTNAME... >&2

source ~/.bashrc
cd ~/Pacman-RL
conda activate pacman

python train_pacman.py --json_path="exp/base/exp_001"

echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"