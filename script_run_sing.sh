#!/bin/bash -l
# Job name
#SBATCH --job-name=DRL_FLEXLAB
#
# Partition:
#SBATCH --partition=lr6
# Constraint:
#SBATCH --constraint=lr6_sky
#
# Wall clock limit:
#SBATCH --time=71:00:00
## Account name
#SBATCH --account=ac_mlee
## QOS
#SBATCH --qos=lr_normal
## to see what modules are available run - module load avail                                                                                                                                 
#SBATCH --mail-type=begin,end,fail                                                                                                                                                           
#SBATCH --mail-user=stouzani@lbl.gov

## Run command                                                                                                                                                                               
module load singularity/3.2.1
echo "module loaded"

singularity exec --bind /global/scratch/stouzani/DRL/DRL_Shed_Tune/DRL_Shed_1/DRL_FLEXLAB:/mnt/shared /global/home/users/stouzani/DRL/new_drl_flexlab_v5.simg ./script_run_hpc.sh
 