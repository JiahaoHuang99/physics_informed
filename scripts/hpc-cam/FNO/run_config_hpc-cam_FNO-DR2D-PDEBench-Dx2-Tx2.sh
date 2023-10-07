#!/bin/bash

# set project
#SBATCH -A TRAFFIC-SL2-GPU

# set partitions
#SBATCH -p ampere

# set max wallclock time
#SBATCH --time=12:00:00

# set name of job
#SBATCH --job-name=FNO-DR2D-PDEBench-Dx2-Tx2

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:1

# set number of CPUs
#SBATCH -c 32

# set size of Memory/RAM
#SBATCH --mem=250G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=huangjiahao0711@gmail.com

# output log
#SBATCH -o /home/jh2446/physics_graph_transformer/physics_informed/server/HPC-CAM/log/FNO-DR2D-PDEBench-Dx2-Tx2.o

# error log
#SBATCH -e /home/jh2446/physics_graph_transformer/physics_informed/server/HPC-CAM/log/FNO-DR2D-PDEBench-Dx2-Tx2.e

. /etc/profile.d/modules.sh

project_path="/home/jh2446/physics_graph_transformer/physics_informed"

# load modules
module purge

module load miniconda/3
module load cuda/11.1
module load cudnn/8.0_cuda-11.1

source activate /home/jh2446/.conda/envs/pde

export WANDB_API_KEY=6bd3bf367138dfcda335e6c5a14e7741a1ea365b

cd ${project_path}

python ${project_path}/train_pino_dr2d_pdebench.py  --config ${project_path}/configs/fno/FNO-DR2D-PDEBench-Dx2-Tx2.yaml --log --device cuda:0

conda deactivate
