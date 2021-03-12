from gym_flexlab.envs import flexlab_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time
from drllib import dqn_functions

from drllib import models, utils

import torch
import torch.optim as optim
import torch.nn as nn

import torch.nn.functional as F

from tensorboardX import SummaryWriter
import time


#LEARNING_RATE = 0.01
#BATCH_SIZE = 8

EPSILON_START = 1.0
EPSILON_STOP = 0.02
EPSILON_STEPS = 5000

REPLAY_BUFFER = 50000


GAMMA = 0.99
BATCH_SIZE = 10 #%64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 200000
REPLAY_INITIAL = 36000

TEST_ITERS = 10# 36000

CUDA = False

RunName = "TestDQN"


class DQN(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


def calc_target(net, local_reward, next_state):
    if next_state is None:
        return local_reward
    state_v = torch.tensor([next_state], dtype=torch.float32)
    next_q_v = net(state_v)
    best_q = next_q_v.max(dim=1)[0].item()
    return local_reward + GAMMA * best_q




def test_net(net, env, writer, frame_idx, count=1, device="cpu"):
    # print(frame_idx)
    # shift = frame_idx + 50000

    dtype=torch.bool

    
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        # t_idx = 0
        while True:
            # t_idx += 1
            obs_v = utils.float32_preprocessor([obs]).to(device)
            print("before")
            
            #net = DQN(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

            mu_v = net(obs_v)
            print("after")

            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)

            # # display actions
            # action_scaled = env.scale_action(action)
            # writer.add_scalar("Actions/SaFr", action_scaled[0], t_idx + shift)
            # writer.add_scalar("Actions/SaTemp", action_scaled[1], t_idx + shift)
            # writer.add_scalar("Actions/CwTemp", action_scaled[2], t_idx + shift)
            # writer.add_scalar("Actions/HwTemp", action_scaled[3], t_idx + shift)
            # writer.add_scalar("Actions/Shade", action_scaled[4], t_idx + shift)
            # writer.add_scalar("Actions/Lig_input", action_scaled[5], t_idx + shift)
            # writer.add_scalar("Actions/P_ctrl", action_scaled[6], t_idx + shift)
            # writer.add_scalar("Actions/PV", action_scaled[7], t_idx + shift)

            # # display observation
            # obs_scaled = env.scale_obs(obs)
            # ZoneTempA_t = obs_scaled[3]
            # writer.add_scalar("Temp/OutTemp_t", obs_scaled[0], t_idx + shift)
            # writer.add_scalar("Temp/OutRH_t", obs_scaled[1], t_idx + shift)
            # writer.add_scalar("Temp/OutSI_t", obs_scaled[1], t_idx + shift)
            # writer.add_scalar("Temp/ZoneTempA_t", obs_scaled[3], t_idx + shift)
            # writer.add_scalar("LigA_t", obs_scaled[4], t_idx + shift)

            obs, reward, done, _ = env.step(action)

            # display change
            # ZoneTempA_t_1 = ZoneTempA_t - obs[3]
            # writer.add_scalar("ZoneTempA_t_1", ZoneTempA_t_1, t_idx + shift)

            # display reward
            # writer.add_scalar("reward_t_test", reward, t_idx + shift)
            # writer.add_scalar("rewards_t_test", rewards, t_idx + shift)

            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def pred_net(net, env, writer, device="cpu"):
    
    buffer = flexlab_env.ExperienceBuffer(env.obs_names, env.action_names)
    rewards = 0.0
    steps = 0
    obs = env.reset()
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
        steps += 1
        if done:
            break
    actions_df = buffer.action_data()
    obs_df = buffer.obs_data()
    reward_df = buffer.reward_data()
    actions_df.to_csv('preds/actions_df.csv',index=False)
    obs_df.to_csv('preds/obs_df.csv',index=False)
    reward_df.to_csv('preds/reward_df.csv',index=False)
    return rewards, steps




if __name__ == "__main__":
   
    device = torch.device("cuda" if CUDA else "cpu")

    save_path = os.path.join("saves", "dqn-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("STARTING PROGRAM")



    env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/FlexlabXR_fmu_2015.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_2015.fmu',
                                 eprice_path = 'e_tariffs/e_price_2015.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2015,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 365,
                                 step_size = 900)


    test_env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/FlexlabXR_fmu_2017.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_2017.fmu',
                                 eprice_path = 'e_tariffs/e_price_2017.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 365,
                                 step_size = 900)


    writer = SummaryWriter(comment="-dqn_" + RunName)

    start_time=time.time()

    print("&&&&&&&&&&&&&&&&&&&&&action_space")
    print(env.action_space.shape[0])
    net = dqn_functions.DQNActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    #target_q = dqn_functions.DQNActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

    #act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)
    target_q = utils.TargetNet(net)


    selector = dqn_functions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = dqn_functions.AgentDQN(net, selector, preprocessor=dqn_functions.float32_preprocessor)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    replay_buffer = utils.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_BUFFER)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    mse_loss = nn.MSELoss()
    print(mse_loss)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    best_reward=None
    print("Before loop")

    while True:
        #print("In true loop")
        step_idx += 1
        selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)
        replay_buffer.populate(1)

        if len(replay_buffer) < BATCH_SIZE:
            continue

        # sample batch
        batch = replay_buffer.sample(BATCH_SIZE)
        batch_states = [exp.state for exp in batch]
        batch_actions = [exp.action for exp in batch]
        batch_targets = [calc_target(net, exp.reward, exp.last_state)
                         for exp in batch]
        # train
        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        net_q_v = net(states_v)
        target_q = net_q_v.data.numpy().copy()
        target_q[range(BATCH_SIZE), batch_actions] = batch_targets
        target_q_v = torch.tensor(target_q)
        loss_v = mse_loss(net_q_v, target_q_v)
        loss_v.backward()
        optimizer.step()
        #print(batch_actions)# actions are changing all the time


        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        
        if new_rewards:
            print("new rewards")
            print(new_rewards)
            print("in loop")
            done_episodes += 1
            reward = new_rewards[0]
            print("reward %s", reward)
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            
            print("%d: reward: %6.2f, mean_100: %6.2f, epsilon: %.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, selector.epsilon, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("epsilon", selector.epsilon, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break
            
            #print time for each episode run
            print("--- %s seconds ---" % (time.time() - start_time))

        newreward=1.2345
        #adding the test code native here
        ################
               

        if step_idx % ((env.n_steps -1) * TEST_ITERS) == 0:
            #net.load_state_dict(target_q.state_dict())
            rewardsq, stepsq = pred_net(net, test_env, writer, device=device)

        ####################
        #testing the dqn
        if step_idx % ((env.n_steps -1) * TEST_ITERS) == 0:
            ts = time.time()
            print("*********************In saving loop")
            #name = "best_%+.3f_%d.dat" % (newreward, step_idx)
            #fname = os.path.join(save_path, name)
            #torch.save(target_q, fname)
            #print("saved")
            #print(fname)




            rewards, steps = test_net(net, test_env, writer, step_idx, device=device)
            print("Test done in %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))


            print("test_reward %s, %s" %(rewards, step_idx))
            print("test_steps %s, %s" %(steps, step_idx))
            print("best_reward and rewards")
            print(best_reward)
            print(rewards)
            if best_reward is None or best_reward < rewards:
                print("saving actual model")
                #if best_reward is not None:
                #best_reward=float(best_reward)
                #rewards=float(rewards)
                #print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                name = "best_%d.dat" % (step_idx)
                fname = os.path.join(save_path, name)
                torch.save(net.state_dict(), fname)
                best_reward = rewards
                print(fname)
            

    writer.close()

