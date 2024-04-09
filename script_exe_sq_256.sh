#!/bin/bash

# Example for serial execution

#SBATCH --partition=prawnew       ## partition (queue) name
#SBATCH --job-name=serial
#SBATCH --output=%x-%j.out      ## %x will be replaced by jobname
#SBATCH --error=%x-%j.err       ## %j will be replaced by jobid
#SBATCH --ntasks=1              ## number of tasks (processes)
##SBATCH --time=1:00:00          ## hour:min:sec

##module load python
##python3 sq_revised.py $1 64 -3.0 2.0 51 4 42
python3 sq_revised_0129.py $1 256 0.01 0.03 101 $2 4 42
