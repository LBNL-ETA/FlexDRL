from gym_flexlab.envs import flexlab_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time

from drllib import models, utils,dqn_functions

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import csv


from tensorboardX import SummaryWriter

CUDA = False

RunName = "Test3"

BESTMODEL = "saves/dqn-TestDQN/best_700780.dat"





def pred_net(net, env, writer, device="cpu"):
    
    buffer = flexlab_env.ExperienceBuffer(env.obs_names, env.action_names)
    rewards = 0.0
    steps = 0
    obs = env.reset()
    reward_list=[]
    while True:
        obs_v = utils.float32_preprocessor([obs]).to(device)
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)

        obs, reward, done, _ = env.step(action)

        action_scaled = env.scale_action(action)
        obs_scaled = env.scale_obs(obs)
        buffer.append(action_scaled,obs_scaled,reward)

        rewards += reward
        reward_list.append(reward)
        steps += 1
        if done:
            break
    actions_df = buffer.action_data()
    actions_df.to_csv('preds/actions_df_dqn20.csv',index=False)

    obs_df = buffer.obs_data()
    obs_df.to_csv('preds/obs_df_dqn20.csv',index=False)
    print(rewards)
    print("final reward")
    print(reward)
    print("reward_list")
    print(reward_list)
    with open('resultfile_dqn20', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(reward_list)

    print("################ SAVING DATA")
    reward_df = buffer.reward_data()
    reward_df.to_csv('preds/reward_df_dqn20.csv',index=False)
    return rewards, steps

if __name__ == "__main__":
    device = torch.device("cuda" if CUDA else "cpu")


    env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/FlexlabXR_fmu_2017.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_2017.fmu',
                                 eprice_path = 'e_tariffs/e_price_2017.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 365,
                                 step_size = 900)

    act_net = dqn_functions.DQNActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    best_model = torch.load(BESTMODEL)
    act_net.load_state_dict(best_model)#load(best_model)
    act_net.train(False)
    act_net.eval()

    writer = SummaryWriter(logdir = "preds", comment="-dqn_" + RunName, max_queue=1,flush_secs=1200)
    
    frame_idx = 0
    best_reward = None
    
    ts = time.time()
    rewards, steps = pred_net(act_net, env, writer, device=device)
    print("Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
    