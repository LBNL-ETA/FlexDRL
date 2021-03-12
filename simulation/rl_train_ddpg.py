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
REPLAY_SIZE = 500000
REPLAY_INITIAL = 150000

TEST_ITERS = 6 # compute test evaluation every 5 episodes

CUDA = False

RunName = "Test5"


def test_net(net, env, writer, exp_idx, count=1, device="cpu"):
    # print(exp_idx)
    # shift = exp_idx + 50000
    
    rewards = 0.0
    e_costs = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        # t_idx = 0
        while True:
            # t_idx += 1
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

    save_path = os.path.join("saves", "ddpg-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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


    test_env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v2/FlexlabXR_v2_SFO_2017.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2017.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2017.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 365,
                                 step_size = 900)

    act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = models.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    # print(act_net)
    # print(crt_net)
    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    writer = SummaryWriter(comment="-ddpg_" + RunName)
    agent = models.AgentDDPG(act_net, device=device)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = utils.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

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
                    print("episodes: %d" % (episodes))


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

                if exp_idx % ((env.n_steps -1) * TEST_ITERS) == 0:
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


