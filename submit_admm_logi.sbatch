#!/bin/bash

#SBATCH --job-name=admm_logi
#SBATCH -p 528_queue
##SBATCH -p skylake
##SBATCH -p general
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=44
#SBATCH --time=60:00:00
#SBATCH --mem=80GB
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=minzhi.hpc.status@gmail.com # send-to address

module purge
module load gsl/2.4
module load openmpi_4.0.1/gcc_9.1.0

work_dir="/proj/yunligrp/users/minzhi/pa2/admm-l1-2-logistic-regression"
cd ${work_dir}

ncore_list=("1" "5" "10" "20" "40" "50" "80")
#ncore_list=("1" "5")
ncore_str=`IFS=',';echo "${ncore_list[*]}";IFS=$' \t\n'`
nrow_list=("10000" "20000" "40000" "100000" "200000" "400000" "1000000" "2000000" "4000000")
#nrow_list=("10000" "20000")
ncol="8000"
#ncol="80"
save_root=".."
reg="l2"

make

for nrow in ${nrow_list[@]}
do
    python data_generation.py ${ncore_str} ${nrow} ${ncol} ${save_root}
    seq_dir="../data_seq_${nrow}_${ncol}"
    
    ./logit -A ${seq_dir}/input.mtx \
    -b ${seq_dir}/output.mtx \
    -e 1e-2 \
    -E 1e-4 \
    -r ${reg} \
    -t 1000 \
    -o ${seq_dir}/solution_seq.dat \
    -p 1 > result_seq_${SLURM_JOB_ID}.out 2>&1
    
    for ncore in ${ncore_list[@]}
    do
        mpi_dir="../data_${nrow}_${ncol}_${ncore}"
        
        mpirun -np ${ncore} ./mpi_logit -A ${mpi_dir}/A \
        -b ${mpi_dir}/b \
        -e 1e-2 \
        -E 1e-4 \
        -r ${reg} \
        -t 1000 \
        -o ${mpi_dir}/solution_mpi_${ncore}.dat \
        -p 1 > result_mpi_${ncore}_${SLURM_JOB_ID}.out 2>&1
    done

done
make clean

exit
