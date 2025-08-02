#!/bin/bash

#OAR -q besteffort
#OAR -l host=1/gpu=1,walltime=3:00:00
#OAR -p kinovis
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

# display some information about attributed resources
# hostname
# nvidia-smi
# module load conda
# # make use of a python torch environment
# conda activate tinyml
# python3 dataset_orchestrator.py
# run the training script
# python3 train.py --epochs 50 --model convnext-large --threshold 1.0 --start_rank 0 --batch_size 32 --worker 8
# Make a loop to run the training script with different start rank (start rank += 2)
for i in {10..50..2}; do
  echo "Running training with start_rank: $i"
  python3 train.py --epochs 50 --threshold 0.2 --start_rank $i --batch_size 32 --worker 8
done
