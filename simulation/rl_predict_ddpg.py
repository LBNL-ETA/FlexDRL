from gym_flexlab.envs import flexlab_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time

from drllib import models, utils

import torch
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

CUDA = False

# ================================
#    Environment HYPERPARAMETERS
# ================================
# RANDOM_SEED = 1234
# Reward function hyperparameters
ALPHA_R = 100.0
BETA_R = 3.0 #1.25
GAMMA_R = 5.0  #1.25
DELTA_R = 1.0
# number of features in the first second NN layers
FEATURES1 = 400
FEATURES2 = 400

# number of hours ahead of eprices
EPRICE_AHEAD = 3
# Number of days that defines an episode
SIM_DAYS = 365
# Number of pv panels
PV_PANELS = 10.0

LIGHT = False

#BESTMODEL = "saves_LRC/saves/DRL_Master/Test_203/best_-70474.384_6307020.dat"
BESTMODEL = "saves_LRC/saves/DRL_Master/Test_203/best_-69177.359_7533385.dat"

def pred_net(net, env, device="cpu"):
    
    buffer = flexlab_env.ExperienceBuffer(env.obs_names, env.action_names)
    rewards = 0.0
    e_costs = 0.0
    steps = 0
    obs = env.reset()
    while True:
        obs_v = utils.float32_preprocessor([obs]).to(device)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)

        obs, reward, done,  divers = env.step(action)
        e_cost = divers[0]
        fan_power = divers[1]
        cool_power = divers[2]
        heat_power = divers[3]
        plugs_load = divers[4]

        action_scaled = env.scale_action(action)
        obs_scaled = env.scale_obs(obs)
        buffer.append(
            action_scaled,
            obs_scaled,
            reward,
            e_cost,
            fan_power,
            cool_power,
            heat_power,
            plugs_load) 

        rewards += reward
        e_costs += e_cost
        steps += 1
        if done:
            break
    actions_df = buffer.action_data()
    obs_df = buffer.obs_data()
    reward_df = buffer.reward_data()
    e_costs_df = buffer.e_cost_data()
    fan_power_df = buffer.fan_power_data()
    cool_power_df = buffer.cool_power_data()
    heat_power_df = buffer.heat_power_data()
    plugs_load_df = buffer.plugs_load_data()

    actions_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/actions_df_ddpg_test_203_2020.csv',index=False)
    obs_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/obs_df_ddpg_test_203_2020.csv',index=False)
    reward_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/reward_df_ddpg_test_203_2020.csv',index=False)
    e_costs_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/e_costs_df_ddpg_test_203_2020.csv',index=False)
    fan_power_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/fan_power_df_ddpg_test_203_2020.csv',index=False)
    cool_power_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/cool_power_df_ddpg_test_203_2020.csv',index=False)
    heat_power_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/heat_power_df_ddpg_test_203_2020.csv',index=False)
    plugs_load_df.to_csv('saves_LRC/saves/DRL_Master/Test_203/plugs_load_df_ddpg_test_203_2020.csv',index=False)

    return rewards, steps, e_costs

if __name__ == "__main__":
    device = torch.device("cuda" if CUDA else "cpu")

    env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_flexlab_2020_new.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_flexlab_2020.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2020_shed_new.csv',
                                 daylight_path= 'daylighting/daylight_SFO_TMY.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2014,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = SIM_DAYS,
                                 step_size = 900,
                                 eprice_ahead = EPRICE_AHEAD,
                                 alpha_r = ALPHA_R,
                                 beta_r = BETA_R,
                                 gamma_r = GAMMA_R,
                                 delta_r = DELTA_R,
                                 pv_panels = PV_PANELS,
                                 light_ctrl = LIGHT)

    # env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2020.fmu',
    #                              battery_path = 'fmu_models/battery.fmu',
    #                              pv_path =  'fmu_models/PV_FMU/PV_SFO_2020.fmu',
    #                              eprice_path = 'e_tariffs/e_d_price_2020_shed.csv',
    #                              daylight_path= 'daylighting/daylight_SFO_2020.csv', 
    #                              chiller_COP = 3.0, 
    #                              boiler_COP = 0.95,
    #                              sim_year = 2014,
    #                              tz_name = 'America/Los_Angeles',
    #                              sim_days = SIM_DAYS,
    #                              step_size = 900,
    #                              eprice_ahead = EPRICE_AHEAD,
    #                              alpha_r = ALPHA_R,
    #                              beta_r = BETA_R,
    #                              gamma_r = GAMMA_R,
    #                              delta_r = DELTA_R,
    #                              pv_panels = PV_PANELS,
    #                              light_ctrl = LIGHT)

    act_net = models.DDPGActor(
        env.observation_space.shape[0], 
        env.action_space.shape[0], 
        FEATURES1, FEATURES2).to(device)

    best_model = torch.load(BESTMODEL)
    act_net.load_state_dict(best_model)
    act_net.train(False)
    act_net.eval()
    
    frame_idx = 0
    best_reward = None
    
    ts = time.time()
    rewards, steps, e_costs = pred_net(act_net, env, device=device)
    print("Test done in %.2f sec, reward %.3f, e_cost %.3f, steps %d" % (time.time() - ts, rewards, e_costs, steps))