#!/bin/bash
#SBATCH --job-name=kaggle-agl
#SBATCH --partition=kill-exclusive
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=0-24:00:00 ## time format is DD-HH:MM:SS

#SBATCH --cpus-per-task=47
#SBATCH --mem=0 ## max amount of memory per node you require
#SBATCH --core-spec=1 ## Uncomment to allow jobs to request all cores on a node    

##SBATCH --gres=gpu:NV-V100-SXM2:2
### To request only 1 of the two GPUs in the node, you would do: gpu:NV-K40:1

#SBATCH --error=t8-%A.err ## %A - filled with jobid
#SBATCH --output=t8-%A.out ## %A - filled with jobid

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=redacted@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html

module purge
module load lang/Anaconda3
module load vis/GLib ## GLIBCXX_3.4.30
source activate ag 

## start tunnel to view ray dashboard in background
##./cloudflared tunnel --url http://127.0.0.1:8265 &

## run autogluon
python kaggle.py 
