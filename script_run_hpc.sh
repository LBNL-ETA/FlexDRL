#!/bin/bash                                                                                                                                                                                  

export PYTHONPATH="/usr/local/JModelica/Python:${PYTHONPATH}"

# python simulation/rl_train_multienv_ddpg.py
python simulation/train_ddpg.py --config_path "simulation/test_configuration/Test_200_configuration.yaml"