from gym_flexlab.envs import flexlab_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time

from drllib import tpro_model, trpo, utils


import torch
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

CUDA = False

RunName = "Test7"

BESTMODEL = "testingtrials/trporun3/best_-2259697.910_700780.dat"#"saves/d4pg-Test6/best_-49936.050_2452730.dat" #best_-4488.278_700780.dat" #best_-124.831_1751950.dat"

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
        # net_energy = divers[1]
        # net_power = divers[2]

        action_scaled = env.scale_action(action)
        obs_scaled = env.scale_obs(obs)
        buffer.append(action_scaled,obs_scaled,reward,e_cost) #,net_energy,net_power)

        rewards += reward
        e_costs += e_cost
        steps += 1
        if done:
            break
    actions_df = buffer.action_data()
    obs_df = buffer.obs_data()
    reward_df = buffer.reward_data()
    e_costs_df = buffer.e_cost_data()
    #net_energy_df = buffer.net_energy_data()
    #net_power_df = buffer.net_power_data()
    actions_df.to_csv('preds/actions_df.csv',index=False)
    obs_df.to_csv('preds/obs_df.csv',index=False)
    reward_df.to_csv('preds/reward_df.csv',index=False)
    e_costs_df.to_csv('preds/e_costs_df.csv',index=False)
    #net_energy_df.to_csv('preds/net_energy_df_ddpg_multienv_002.csv',index=False)
    #net_power_df.to_csv('preds/net_power_df_ddpg_multienv_002.csv',index=False)

    return rewards, steps, e_costs

if __name__ == "__main__":
    device = torch.device("cuda" if CUDA else "cpu")

    env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v2/FlexlabXR_v2_SFO_2015.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2015.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2015.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2015,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 365,
                                 step_size = 900)

    act_net = tpro_model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    print(act_net)

    best_model = torch.load(BESTMODEL)
    act_net.load_state_dict(best_model)
    print("1")
    act_net.train(False)
    act_net.eval()
    
    frame_idx = 0
    best_reward = None
    
    ts = time.time()
    rewards, steps, e_costs = pred_net(act_net, env, device=device)
    print("Test done in %.2f sec, reward %.3f, e_cost %.3f, steps %d" % (time.time() - ts, rewards, e_costs, steps))

