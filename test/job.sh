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

# for rank in 8 16 32 64 128 256; do
#     for nleaf in 512 1024 2048; do
#         ./hicmat-matern $nleaf $rank 16384
#     done
# done

./hlu-laplace3d 2048 1000 16384 1

for fname in *xml; do
    python ../visualization.py $fname
done
