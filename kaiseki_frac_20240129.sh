#!/bin/bash

# Example for serial execution

#SBATCH --partition=prawnew       ## partition (queue) name
#SBATCH --job-name=serial
#SBATCH --output=%x-%j.out      ## %x will be replaced by jobname
#SBATCH --error=%x-%j.err       ## %j will be replaced by jobid
#SBATCH --ntasks=1              ## number of tasks (processes)
##SBATCH --time=1:00:00          ## hour:min:sec

##module load python
python3 kaiseki_frac_20240129.py
