#!/bin/sh

root_dir=$1

mkdir ${root_dir}/Containers
mkdir ${root_dir}/Simages

singularity build --sandbox /global/scratch/stouzani/Containers/drl_flexlab docker://stouzani/drl_flexlab:1
singularity build /global/scratch/stouzani/Simages/drl_flexlab.simg /global/scratch/stouzani/Containers/drl_flexlab



from gym_flexlab.envs import flexlab_env
envB = flexlab_env.EnvsBatch(envelope_dir = 'fmu_models/EPlus_FMU_v1/',battery_path = 'fmu_models/battery.fmu',pv_dir =  'fmu_models/PV_FMU/',eprice_path = 'e_tariffs/e_price_2015.csv')
env=envB.sample_env()
env.stop_fmu()
envB.stop_env()

import os
os.remove("battery_log.txt")
os.remove("PV_log.txt")
os.remove("FlexlabXR_v1_Eco_log.txt")