#!/bin/bash -l                                                                                                                                                                                              
# Job name                                                                                                                                                                                                  
#SBATCH --job-name=DRL_FLEXLAB                                                                                                                                                                                   
#                                                                                                                                                                                                           
# Constraint:                                                                                                                                                                                               
#SBATCH --constraint=es1_v100                                                                                                                                                                              
# QoS:                                                                                                                                                                                                      
#SBATCH --qos=es_normal                                                                                                                                                                                     
#                                                                                                                                                                                                           
# Account:                                                                                                                                                                                                  
#SBATCH --account=pc_mlee                                                                                                                                                                                   
#                                                                                                                                                                                                           
# Requeue:                                                                                                                                                                                                  
##SBATCH --requeue                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
#                                                                                                                                                                                                           
# CPU cores:                                                                                                                                                                                                
#SBATCH --cpus-per-task=6                                                                                                                                                                                   
#                                                                                                                                                                                                           
# Wall clock limit:                                                                                                                                                                                         
#SBATCH --time=71:55:30                                                                                                                                                                                        
#                                                                                                                                                                                                           
# Mail type:                                                                                                                                                                                                
#SBATCH --mail-type=all                                                                                                                                                                                     
#                                                                                                                                                                                                           
# Mail user:                                                                                                                                                                                                
#SBATCH --mail-user=stouzani@lbl.gov                                                                                                                                                                        
#                                                                                                                                                                                                           
# Error file:                                                                                                                                                                                               
##SBATCH --error=std3.err                                                                                                                                                                                   
## Run command                                                                                                                                                                                              

module load singularity/3.2.1
echo "module loaded"
singularity exec --bind /global/scratch/stouzani/DRL/DRL_FLEXLAB:/mnt/shared  /global/scratch/stouzani/DRL/drl_flexlab_1.sif cd /mnt/shared && pwd



/global/home/users/stouzani/DRL/DRL_FLEXLAB

singularity run --bind /global/home/users/stouzani/DRL/DRL_FLEXLAB:/mnt/shared  /global/home/users/stouzani/DRL/drl_flexlab_1.sif cd /mnt/shared && pwd


singularity run --bind `pwd`:/mnt/shared  /global/scratch/stouzani/Simages/drl_flexlab.simg cd /mnt/shared && pwd





# Partition:
#SBATCH --partition=cf1
#
# Wall clock limit:
#SBATCH --time=0:20:30
### how to see what partition,qos and account you have access to
### run this command - sacctmgr show association user=mkiran -p
## Account name
#SBATCH --account=pc_mlee
## QOS
#SBATCH --qos=cf_normal
# Command
## Load the module first
## to see what modules are available run - module load avail
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=mkiran@lbl.gov

## Run command
module load singularity/3.2.1
echo "module loaded"

#singularity run --bind /global/home/users/mkiran/ETA/DRL_FLEXLAB:/mnt/shared new_drl_flexlab.simg


singularity exec --bind /global/home/users/stouzani/DRL/DRL_FLEXLAB:/mnt/shared new_drl_flexlab.simg python simulation/rl_train_ddpg.py


#module load python/2.7
## Now run your command, this is your code which you have copied to your Lawrencium Directory.
#python simulation/rl_train_ddpg.py
#####END COPYING HERE######