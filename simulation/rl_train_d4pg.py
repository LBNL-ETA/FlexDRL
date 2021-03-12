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
REPLAY_INITIAL = 250000
REWARD_STEPS = 3

TEST_ITERS = 5 # compute test evaluation every 5 episodes

Vmax = 100.0
Vmin = -100.0
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
CUDA = False

RunName = "Test6"

def test_net(net, env, writer, frame_idx, count=1, device="cpu"):
    # print(frame_idx)
    # shift = frame_idx + 50000
    
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

def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device="cpu"):
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    # parser.add_argument("-n", "--name", required=True, help="Name of the run")
    # args = parser.parse_args()
    device = torch.device("cuda" if CUDA else "cpu")

    save_path = os.path.join("saves", "d4pg-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    act_net = models.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = models.D4PGCritic(env.observation_space.shape[0], env.action_space.shape[0], N_ATOMS, Vmin, Vmax).to(device)
    # print(act_net)
    # print(crt_net)
    tgt_act_net = utils.TargetNet(act_net)
    tgt_crt_net = utils.TargetNet(crt_net)

    writer = SummaryWriter(comment="-d4pg_" + RunName)
    agent = models.AgentDDPG(act_net, device=device)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    buffer = utils.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    episodes = 0
    with utils.RewardTracker(writer) as tracker:
        with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)
                    episodes += 1
                    print("episodes: %d" % (episodes))

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE) # sample the training batch
                states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch_ddqn(batch, device)

                # train critic
                crt_opt.zero_grad()
                crt_distr_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                last_distr_v = F.softmax(tgt_crt_net.target_model(last_states_v, last_act_v), dim=1)
                proj_distr_v = distr_projection(last_distr_v, rewards_v, dones_mask,
                                                gamma=GAMMA**REWARD_STEPS, device=device)
                prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
                critic_loss_v = prob_dist_v.sum(dim=1).mean()
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                crt_distr_v = crt_net(states_v, cur_actions_v)
                actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % ((env.n_steps -1) * TEST_ITERS) == 0:
                    ts = time.time()
                    rewards, steps, e_costs = test_net(act_net, test_env, writer, frame_idx, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_e_costs", e_costs, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards

    pass
