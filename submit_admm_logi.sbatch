#!/bin/bash

#SBATCH --job-name=admm_logi
#SBATCH -p 528_queue
##SBATCH -p skylake
##SBATCH -p general
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=44
#SBATCH --time=00:20:00
#SBATCH --mem=30GB
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=minzhi.hpc.status@gmail.com # send-to address

module purge
module load gsl/2.4
module load openmpi_4.0.1/gcc_9.1.0
#module load openmpi/3.0.0/gnu/4.8.5/local_openmpi

work_dir="/proj/yunligrp/users/minzhi/admm-l1-2-logistic-regression"
cd ${work_dir}

ncores=80
nrow=4000000
ncol=8000
python data_generation.py ${ncores} ${nrow} ${ncores}

make
mpirun -np ${ncores} ./mpi_logit > result_${SLURM_JOB_NAME}.out
make clean

exit
