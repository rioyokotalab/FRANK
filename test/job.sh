#!/bin/bash

#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -N HICMATEST
#$ -o HICMATEST_out.log
#$ -e HICMATEST_err.log

. /etc/profile.d/modules.sh
module purge
module load intel-mkl
source ~/.bashrc

export MKL_NUM_THREADS=1

for rank in 8 16 32 64 128 256; do
    ./hicmat-matern 1024 $rank 65536
done
