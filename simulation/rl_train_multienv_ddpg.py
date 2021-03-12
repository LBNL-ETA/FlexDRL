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


# ================================
#    TRAINING HYPERPARAMETERS
# ================================
# The used docker container does not allow CUDA optimization
CUDA = False
# Name of the experiment
RUN_NAME = "Test110"
# Number of episodes before test evaluation
TEST_ITERS = 5
# Actor learning rates
ACTOR_LEARNING_RATE = 0.00005
# Critic earning rates  and 
CRITIC_LEARNING_RATE = 0.0001
# Maximum number of episodes
MAX_EPISODES = 1000
# Discount factor
GAMMA = 0.98
# Size of replay buffer
REPLAY_BUFFER_SIZE = 1500000
# Minimum transitions in the buffer before start learning
REPLAY_INITIAL = 1500000
# Training batch size
BATCH_SIZE = 128
# Exploration gaussian noise variables
EPSILON = 0.3
# Ornstein-Uhlenbeck variables
OU_ENABLE = True
OU_THETA = 0.15
OU_MU = 0.0
OU_SIGMA = 0.3
# Exploration duration
EXPLORATION_TIME = 200
# number of features in the first second NN layers
FEATURES1 = 400
FEATURES2 = 300

# ================================
#    Environment HYPERPARAMETERS
# ================================
# RANDOM_SEED = 1234
# Reward function hyperparameters
ALPHA_R = 100.0
BETA_R = 1.0
GAMMA_R = 1.0
DELTA_R = 15.0
# number of hours ahead of eprices
EPRICE_AHEAD = 2
# Number of days that defines an episode
SIM_DAYS = 365
# Number of pv panels
PV_PANELS = 10.0



def test_net(net, env, writer, exp_idx, count=1, device="cpu"):
    
    rewards = 0.0
    e_costs = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, divers = env.step(action)
            e_cost = divers[0]
            rewards += reward
            e_costs += e_cost
            steps += 1
            if done:
                break
    return rewards / count, steps / count, e_costs / count


if __name__ == "__main__":
    device = torch.device("cuda" if CUDA else "cpu")

    save_path = os.path.join("saves", "ddpg-" + RUN_NAME)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    env1 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2013.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2013.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2013.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2013.csv',
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2013,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = SIM_DAYS,
                                 step_size = 900,
                                 eprice_ahead = EPRICE_AHEAD,
                                 alpha_r = ALPHA_R,
                                 beta_r = BETA_R,
                                 gamma_r = GAMMA_R,
                                 delta_r = DELTA_R,
                                 pv_panels = PV_PANELS)

    env2 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2014.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2014.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2014.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2014.csv', 
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
                                 pv_panels = PV_PANELS)

    env3 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2015.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2015.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2015.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2015.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2015,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = SIM_DAYS,
                                 step_size = 900,
                                 eprice_ahead = EPRICE_AHEAD,
                                 alpha_r = ALPHA_R,
                                 beta_r = BETA_R,
                                 gamma_r = GAMMA_R,
                                 delta_r = DELTA_R,
                                 pv_panels = PV_PANELS)

    env4 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_TMY.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_TMY.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2017.csv',
                                 daylight_path= 'daylighting/daylight_SFO_TMY.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = SIM_DAYS,
                                 step_size = 900,
                                 eprice_ahead = EPRICE_AHEAD,
                                 alpha_r = ALPHA_R,
                                 beta_r = BETA_R,
                                 gamma_r = GAMMA_R,
                                 delta_r = DELTA_R,
                                 pv_panels = PV_PANELS)

    # env5 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_Miami_TMY.fmu',
    #                              battery_path = 'fmu_models/battery.fmu',
    #                              pv_path =  'fmu_models/PV_FMU/PV_Miami_TMY.fmu',
    #                              eprice_path = 'e_tariffs/e_d_price_2017.csv',
    #                              daylight_path= 'daylighting/daylight_Miami_TMY.csv', 
    #                              chiller_COP = 3.0, 
    #                              boiler_COP = 0.95,
    #                              sim_year = 2017,
    #                              tz_name = 'America/Los_Angeles',
    #                              sim_days = SIM_DAYS,
    #                              step_size = 900,
    #                              eprice_ahead = EPRICE_AHEAD,
    #                              alpha_r = ALPHA_R,
    #                              beta_r = BETA_R,
    #                              gamma_r = GAMMA_R,
    #                              delta_r = DELTA_R,
    #                              pv_panels = PV_PANELS)

    # env6 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_Chicago_TMY.fmu',
    #                              battery_path = 'fmu_models/battery.fmu',
    #                              pv_path =  'fmu_models/PV_FMU/PV_Chicago_TMY.fmu',
    #                              eprice_path = 'e_tariffs/e_d_price_2017.csv',
    #                              daylight_path= 'daylighting/daylight_Chicago_TMY.csv', 
    #                              chiller_COP = 3.0, 
    #                              boiler_COP = 0.95,
    #                              sim_year = 2017,
    #                              tz_name = 'America/Los_Angeles',
    #                              sim_days = SIM_DAYS,
    #                              step_size = 900,
    #                              eprice_ahead = EPRICE_AHEAD,
    #                              alpha_r = ALPHA_R,
    #                              beta_r = BETA_R,
    #                              gamma_r = GAMMA_R,
    #                              delta_r = DELTA_R,
    #                              pv_panels = PV_PANELS)    


    test_env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2017.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2017.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2017.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2017.csv',  
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = SIM_DAYS,
                                 step_size = 900,
                                 eprice_ahead = EPRICE_AHEAD,
                                 alpha_r = ALPHA_R,
                                 beta_r = BETA_R,
                                 gamma_r = GAMMA_R,
                                 delta_r = DELTA_R,
                                 pv_panels = PV_PANELS)

    env =[env1,env2,env3,env4]#,env5,env6]

    act_net = models.DDPGActor(
        env1.observation_space.shape[0], 
        env1.action_space.shape[0], 
        FEATURES1, FEATURES2).to(device)
    crt_net = models.DDPGCritic(
        env1.observation_space.shape[0], 
        env1.action_space.shape[0], 
        FEATURES1, FEATURES2).to(device)
    # print(act_net)
    # print(crt_net)
    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + RUN_NAME)
    agent = models.AgentDDPG(act_net,
                             device = device, 
                             ou_enabled = OU_ENABLE,
                             ou_mu = OU_MU,
                             ou_teta = OU_THETA,
                             ou_sigma = OU_SIGMA, 
                             epsilon = EPSILON)

    env_i = random.choice(env)

    print("#######################################")
    print("#######################################")
    print("#######################################")
    print("env_i:")
    print((env_i.pv_path))
    print("#######################################")
    print("#######################################")
    print("#######################################")

    exp_source = utils.ExperienceSourceFirstLast(env_i, agent, gamma=GAMMA, steps_count=1)
    buffer = utils.ExperienceReplayBufferMultiEnv(buffer_size=REPLAY_BUFFER_SIZE)
    buffer.set_exp_source(exp_source)
    
    act_opt = optim.Adam(act_net.parameters(), lr=ACTOR_LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=CRITIC_LEARNING_RATE)

    exp_idx = 0
    best_reward = None
    episodes = 0
    with utils.RewardTracker(writer) as tracker:
        with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                exp_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
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
                    print("#######################################")
                    print("#######################################")
                    print("#######################################")
                    print("env_i:")
                    print((env_i.pv_path))
                    print("#######################################")
                    print("#######################################")
                    print("#######################################")
                    exp_source = utils.ExperienceSourceFirstLast(env_i, agent, gamma=GAMMA, steps_count=1)
                    buffer.set_exp_source(exp_source)


                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                #print("q_v")
                #print(q_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, exp_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), exp_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, exp_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if exp_idx % ((env_i.n_steps -1) * TEST_ITERS) == 0:
                    ts = time.time()
                    rewards, steps, e_costs = test_net(act_net, test_env, writer, exp_idx, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, exp_idx)
                    writer.add_scalar("test_e_costs", e_costs, exp_idx)
                    writer.add_scalar("test_steps", steps, exp_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, exp_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    pass


