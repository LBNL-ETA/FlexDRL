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


class Predict:
    def __init__(self, out_dir_path, config_file="test_config/test_configuration.yaml"):

        with open(config_file, 'r') as fp:
            self.config = yaml.safe_load(fp)

        # Name of the experiment
        self.test_name = self.config.get("test_name")
        self.out_path = os.path.join(out_dir_path, self.test_name)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

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

        pred_env = environments.set_pred_env(self.sim_days, self.eprice_ahead, self.alpha_r, self.beta_r, self.gamma_r, self.pv_panels, self.light_ctrl)

        self.pred_env = pred_env

    def prediction(self,net=None):
        buffer = flexlab_env.ExperienceBuffer(self.pred_env.obs_names, self.pred_env.action_names)
        rewards = 0.0
        e_costs = 0.0
        steps = 0
        obs = self.pred_env.reset()
        while True:
            obs_v = utils.float32_preprocessor([obs]).to(self.device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done,  divers = self.pred_env.step(action)
            e_cost = divers[0]
            fan_power = divers[1]
            cool_power = divers[2]
            heat_power = divers[3]
            plugs_load = divers[4]

            action_scaled = self.pred_env.scale_action(action)
            obs_scaled = self.pred_env.scale_obs(obs)
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
        final_df = pd.concat([actions_df,obs_df,reward_df,e_costs_df,fan_power_df,cool_power_df,heat_power_df,plugs_load_df],axis=1)
        final_df.to_csv(os.path.join(self.out_path, "prediction_results.csv"),index=False)

        return rewards, steps, e_costs




class PredDDPG(Predict):
    def __init__(self, out_dir_path, best_model, config_file="test_config/test_configuration.yaml"):
        Predict.__init__(self, out_dir_path, config_file)
        self.nn_features1 = self.config.get("nn_features1") # number of features in the first NN layers
        self.nn_features2 = self.config.get("nn_features2") # number of features in the second NN layers
        self.act_net = models.DDPGActor(
            self.pred_env.observation_space.shape[0], 
            self.pred_env.action_space.shape[0], 
            self.nn_features1, self.nn_features2).to(self.device)
        self.best_model = torch.load(best_model)
        self.act_net.load_state_dict(self.best_model)
        self.act_net.train(False)
        self.act_net.eval()

    def prediction(self):
        net = self.act_net
        Predict.prediction(self,net)


