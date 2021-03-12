import os
import pytz
import random
import time
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import environments
from datetime import timedelta
from drllib import models, utils
from tensorboardX import SummaryWriter
from gym_flexlab.envs import flexlab_env


class Train:
    def __init__(self,config_file="test_config/test_configuration.yaml"):

        with open(config_file, 'r') as fp:
            self.config = yaml.safe_load(fp)

        # Name of the experiment
        self.test_name = self.config.get("test_name")

        # Flexlab gym environement parameters
        #self.config = self.flexlab_config.get("flexlab_parameters")
        self.light_ctrl = self.config.get("light_ctrl")
        self.sim_days = self.config.get("episode_simulation_days")
        self.pv_panels = self.config.get("pv_panels_number")
        self.alpha_r = self.config.get("alpha_reward")
        self.beta_r = self.config.get("beta_reward")
        self.gamma_r = self.config.get("gamma_reward")
        self.eprice_ahead = self.config.get("eprice_ahead")

        self.cuda = False # no cuda available with the used docker container
        self.device = torch.device("cuda" if self.cuda else "cpu")

        env, test_env = environments.set_envs(self.sim_days, self.eprice_ahead, self.alpha_r, self.beta_r, self.gamma_r, self.pv_panels, self.light_ctrl)

        self.env = env
        self.test_env = test_env


class TrainDDPG(Train):
    def __init__(self, config_file="test_config/test_configuration.yaml"):
        Train.__init__(self,config_file)

        # Training hyperparameters configuration
        #self.train_config = self.config.get("ddpg_training_hyperparameters")
        self.test_iters = self.config.get("test_iterations") # Number of episodes before test evaluation
        self.actor_lr = self.config.get("actor_learning_rate")
        self.critic_lr = self.config.get("critic_learning_rate")
        self.gamma = self.config.get("gamma")
        self.buffer_size = self.config.get("replay_buffer_size")
        self.init_buffer_size = self.config.get("initial_replay_buffer_size")
        self.batch_size = self.config.get("batch_size")
        self.ou_enable = self.config.get("ou_enable") # Ornstein-Uhlenbeck
        self.ou_theta = self.config.get("ou_theta") # Ornstein-Uhlenbeck
        self.ou_sigma = self.config.get("ou_sigma") # Ornstein-Uhlenbeck
        self.ou_mu = self.config.get("ou_mu") # Ornstein-Uhlenbeck
        self.epsilon = self.config.get("epsilon") # exploration gaussian noise variables
        self.nn_features1 = self.config.get("nn_features1") # number of features in the first NN layers
        self.nn_features2 = self.config.get("nn_features2") # number of features in the second NN layers

        self.save_path = os.path.join("saves", "ddpg-" + self.test_name)
        if not os.path.exists( self.save_path):
            os.makedirs( self.save_path)

        self.act_net = models.DDPGActor(
            self.test_env.observation_space.shape[0], 
            self.test_env.action_space.shape[0], 
            self.nn_features1, self.nn_features2).to(self.device)

        self.crt_net = models.DDPGCritic(
            self.test_env.observation_space.shape[0], 
            self.test_env.action_space.shape[0], 
            self.nn_features1, self.nn_features2).to(self.device)

        self.tgt_act_net = utils.TargetNet(self.act_net)
        self.tgt_crt_net = utils.TargetNet(self.crt_net)

        self.writer = SummaryWriter(comment="-ddpg_" + self.test_name)

        self.agent = models.AgentDDPG(self.act_net,
                                      device = self.device, 
                                      ou_enabled = self.ou_enable,
                                      ou_mu = self.ou_mu,
                                      ou_teta = self.ou_theta,
                                      ou_sigma = self.ou_sigma, 
                                      epsilon = self.epsilon)

    def training(self):

        env_i = random.choice(self.env)
        print("#######################################")
        print("#######################################")
        print("The used environment is: {0}".format(env_i.pv_path))
        print("#######################################")
        print("#######################################")

        exp_source = utils.ExperienceSourceFirstLast(env_i, self.agent, gamma=self.gamma, steps_count=1)
        buffer = utils.ExperienceReplayBufferMultiEnv(buffer_size=self.buffer_size)
        buffer.set_exp_source(exp_source)
        
        act_opt = optim.Adam(self.act_net.parameters(), lr=self.actor_lr)
        crt_opt = optim.Adam(self.crt_net.parameters(), lr=self.critic_lr)

        exp_idx = 0
        best_reward = None
        episodes = 0

        with utils.RewardTracker(self.writer) as tracker:
            with utils.TBMeanTracker(self.writer, batch_size=10) as tb_tracker:
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
                        print("episodes: %d" % (episodes))
                        print("#######################################")
                        print("#######################################")

                        env_i = random.choice(self.env)

                        print("#######################################")
                        print("#######################################")
                        print("The used environment is: {0}".format(env_i.pv_path))
                        print("#######################################")
                        print("#######################################")

                        exp_source = utils.ExperienceSourceFirstLast(env_i, self.agent, gamma=self.gamma, steps_count=1)
                        buffer.set_exp_source(exp_source)


                    if len(buffer) < self.init_buffer_size:
                        continue

                    batch = buffer.sample(self.batch_size)
                    states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch_ddqn(batch, self.device)

                    # train critic
                    crt_opt.zero_grad()
                    q_v = self.crt_net(states_v, actions_v)
                    last_act_v = self.tgt_act_net.target_model(last_states_v)
                    q_last_v = self.tgt_crt_net.target_model(last_states_v, last_act_v)
                    q_last_v[dones_mask] = 0.0
                    q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * self.gamma
                    critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                    critic_loss_v.backward()
                    crt_opt.step()
                    tb_tracker.track("loss_critic", critic_loss_v, exp_idx)
                    tb_tracker.track("critic_ref", q_ref_v.mean(), exp_idx)

                    # train actor
                    act_opt.zero_grad()
                    cur_actions_v = self.act_net(states_v)
                    actor_loss_v = -self.crt_net(states_v, cur_actions_v)
                    actor_loss_v = actor_loss_v.mean()
                    actor_loss_v.backward()
                    act_opt.step()
                    tb_tracker.track("loss_actor", actor_loss_v, exp_idx)

                    self.tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                    self.tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                    if exp_idx % ((env_i.n_steps -1) * self.test_iters) == 0:
                        ts = time.time()
                        rewards, steps, e_costs = test_net(self.act_net, self.test_env, self.writer, exp_idx, device=self.device)
                        print("#######################################")
                        print("#######################################")
                        print("Test done in %.2f sec, reward %.3f, steps %d" % (time.time() - ts, rewards, steps))
                        print("#######################################")
                        print("#######################################")
                        self.writer.add_scalar("test_reward", rewards, exp_idx)
                        self.writer.add_scalar("test_e_costs", e_costs, exp_idx)
                        self.writer.add_scalar("test_steps", steps, exp_idx)
                        if best_reward is None or best_reward < rewards:
                            if best_reward is not None:
                                print("#######################################")
                                print("#######################################")
                                print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                                print("#######################################")
                                print("#######################################")
                                name = "best_%+.3f_%d.dat" % (rewards, exp_idx)
                                fname = os.path.join(self.save_path, name)
                                torch.save(self.act_net.state_dict(), fname)
                            best_reward = rewards

        pass

def test_net(net, env, writer, exp_idx, count=1, device="cpu"):
    
    rewards = 0.0
    e_costs = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = utils.float32_preprocessor([obs]).to(device)
            action_v = net(obs_v)
            action = action_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, divers = env.step(action)
            e_cost = divers[0]
            rewards += reward
            e_costs += e_cost
            steps += 1
            if done:
                break
    return rewards / count, steps / count, e_costs / count




