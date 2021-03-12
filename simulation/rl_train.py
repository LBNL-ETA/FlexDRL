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


GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 50
REPLAY_INITIAL = 5#250000

TEST_ITERS = 1 # compute test evaluation every 5 episodes

CUDA = False

RunName = "Test5"


if __name__ == "__main__":
    device = torch.device("cuda" if CUDA else "cpu")


    env1 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v1/FlexlabXR_v1_SFO_2015.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2015.fmu',
                                 eprice_path = 'e_tariffs/price2013_2020.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2015,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 5,
                                 step_size = 900)


    env2 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v1/FlexlabXR_v1_SFO_2017.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2017.fmu',
                                 eprice_path = 'e_tariffs/price2013_2020.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 5,
                                 step_size = 900)

    env3 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v1/FlexlabXR_v1_SFO_2013.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2013.fmu',
                                 eprice_path = 'e_tariffs/price2013_2020.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2013,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 5,
                                 step_size = 900)
    
    env4 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v1/FlexlabXR_v1_SFO_2014.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2014.fmu',
                                 eprice_path = 'e_tariffs/price2013_2020.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2014,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 5,
                                 step_size = 900)

    env5 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v1/FlexlabXR_v1_SFO_TMY.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_TMY.fmu',
                                 eprice_path = 'e_tariffs/price2013_2020.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 5,
                                 step_size = 900)

    env6 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v1/FlexlabXR_v1_flexlab_2018.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_flexlab_2018.fmu',
                                 eprice_path = 'e_tariffs/price2013_2020.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2018,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 5,
                                 step_size = 900)

    env =[env1,env2,env3,env4,env5]#,env6]

    act_net = models.DDPGActor(env1.observation_space.shape[0], env1.action_space.shape[0]).to(device)
    crt_net = models.DDPGCritic(env1.observation_space.shape[0], env1.action_space.shape[0]).to(device)
    # print(act_net)
    # print(crt_net)
    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_")
    agent = models.AgentDDPG(act_net, device=device)

    exp_source = utils.ExperienceSourceFirstLast(env1, agent, gamma=GAMMA, steps_count=1)
    buffer = utils.ExperienceReplayBufferMultiEnv(buffer_size=REPLAY_SIZE)
    buffer.set_exp_source(exp_source)

    exp_idx = 0
    best_reward = None
    episodes = 0
    with utils.RewardTracker(writer) as tracker:
        with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                exp_idx += 1
                buffer.populate(1)
                
                rewards_steps = exp_source.pop_rewards_steps()
                # print("rewards_steps")
                # print(rewards_steps)
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], exp_idx)
                    tracker.reward(rewards[0], exp_idx)
                    episodes += 1
                    print("#######################################")
                    print("#######################################")
                    print("#######################################")
                    print("episodes: %d" % (episodes))
                    print("#######################################")
                    print("#######################################")
                    print("#######################################")
                    env_i = random.choice(env)
                    exp_source = utils.ExperienceSourceFirstLast(env_i, agent, gamma=GAMMA, steps_count=1)
                    buffer.set_exp_source(exp_source)



                if len(buffer) < REPLAY_INITIAL:
                    print("len(buffer)")
                    print(len(buffer))
                    # print("##########buffer.buffer##########")
                    # print(buffer.buffer)
                    continue

    pass


