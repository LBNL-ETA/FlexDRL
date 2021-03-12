# -*- coding: utf-8 -*-
"""

"""

from gym_flexlab.envs import flexlab_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import time
from drllib import models, utils
from flexlab.db_layer import db_interface
import torch
import torch.optim as optim
import torch.nn.functional as F


class FlexlabController:
    """
    Controller interface with the DRL model
    """
    def __init__(
        self,
        best_model_path = None,
        eprice_ahead = 3,
        eprice_path = None,
        device = "cpu",
        nn_features_1 = 400,
        nn_features_2 = 400
        ):

        self.eprice_path = eprice_path
        self.eprice_df = pd.read_csv(self.eprice_path, index_col=0, parse_dates=True)

        self.eprice_ahead = eprice_ahead

        # Define observation space

        self.main_obs_names = [
            "TOD", "OutTemp", "OutRH", "OutSI",
            "ZoneTempB", "net_power","net_energy",
            "SOC", "PV"]

        eprice_24_ahead =[
            "eprice_t0", "eprice_t1","eprice_t2",
            "eprice_t3", "eprice_t4", "eprice_t5",
            "eprice_t6", "eprice_t7", "eprice_t8",
            "eprice_t9", "eprice_t10", "eprice_t11",
            "eprice_t12", "eprice_t13", "eprice_t14",
            "eprice_t15", "eprice_t16", "eprice_t17",
            "eprice_t18", "eprice_t19", "eprice_t20",
            "eprice_t21", "eprice_t22", "eprice_t23"]

        eprice_x_ahead = eprice_24_ahead[0:eprice_ahead] 

        obs_names = self.main_obs_names + eprice_x_ahead

        self.obs_names = obs_names
        
        min_eprice = min(self.eprice_df['e_price'])
        max_eprice = max(self.eprice_df['e_price'])

        main_obs_low = np.array([
            0.0, -20.0, 0.0, 0.0, 
            5.0, -5.0, -1.25, 
            0.0, 0.0])
        
        main_obs_high = np.array([
            23.75, 40.0, 100.0, 20000.0,
            35.0, 20.0, 5.0,
            1.0, 5000.0])
        
        eprice_x_ahead_low = np.repeat(min_eprice, eprice_ahead)
        eprice_x_ahead_high = np.repeat(max_eprice, eprice_ahead)

        obs_space_low = (main_obs_low,eprice_x_ahead_low)
        self.obs_space_low = np.concatenate(obs_space_low)
        
        obs_space_high = (main_obs_high,eprice_x_ahead_high)
        self.obs_space_high = np.concatenate(obs_space_high)

        # Define action space

        self.action_names = ["SaFrB", "SaTempB", "P_ctrl"]
        self.action_init = np.array([0.5, 22.0, 0.0])
        self.action_space_low = np.array([0.05, 10.0, -3300.0])
        self.action_space_high = np.array([2.0, 30.0, 3300.0])

        # Initiate Flexlab DB
        self.db_flexlab = db_interface.DB_Interface()

        # load DRL net

        self.best_model_path = best_model_path

        self.act_net = models.DDPGActor(
            len(self.obs_names),
            len(self.action_names), 
            nn_features_1, nn_features_2).to(device)

        best_model = torch.load(self.best_model_path)
        self.act_net.load_state_dict(best_model)
        self.act_net.train(False)
        self.act_net.eval()



    def predict_action(self, now_t,device="cpu"):
        
        obs = self._get_obs(now_t)
        obs_scaled = (2.0 * (obs - self.obs_space_low) / \
                (self.obs_space_high - self.obs_space_low) - 1)
        print("obs")
        print(self.obs_names)
        print(obs)
        print("obs_scaled")
        print(obs_scaled)
        obs_v = utils.float32_preprocessor([obs_scaled]).to(device)
        mu_v = self.act_net(obs_v)
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        print("action_before")
        print(action)
        action = np.clip(action, -1, 1)
        
        action_rescaled = self._scale_action(action)
        print("action_after")
        print(action_rescaled)
        action_dict = {
            'sup_air_flow_sp': [action_rescaled[self.action_names.index("SaFrB")]],
            'sup_air_temp_sp': [action_rescaled[self.action_names.index("SaTempB")]],
            'battery_sp': [action_rescaled[self.action_names.index("P_ctrl")]]}
        self.action_df = pd.DataFrame.from_dict(action_dict)

        return self.action_df

    
    def push_actions(self, actions):
        return self.db_flexlab.push_setpoints_to_db(cell = 'b', df = actions)




    
    def _get_obs(self, now_t, step_size = 15):
        end_t = now_t
        start_t = now_t - timedelta(minutes = step_size - 1)
        obs_df =  self.db_flexlab.get_data_controller(st = start_t, et = end_t).reset_index()

        #obs_df = obs_df.reset_index()

        # print(start_t)

        # print(obs_df)

        obs = []

        for obs_name in self.main_obs_names:
            obs.append(obs_df.loc[[0],obs_name].values[0])



        if self.eprice_ahead != 0:
            eprices_ahead = self._eprice_ahead(self.eprice_ahead, now_t = now_t)
            
            obs = (obs, eprices_ahead)
            obs = np.concatenate(obs)
            
        # print("obs")
        # print(obs)
        return obs


    def _eprice_ahead(self, hours_ahead, now_t):
        times_ahead = hours_ahead * 60
        start_t = now_t
        end_t = start_t + timedelta(minutes = times_ahead-15)


        df = get_data_from_csv(
            path = self.eprice_path, 
            start_time = start_t, 
            end_time = end_t)
        # eprice_ahead = df.groupby([df.index.hour], sort=False)['e_price'].mean()
        eprice_ahead = []
        for i in range(0,df.shape[0],4):
            eprice_ahead.append(df.iloc[i]["e_price"])

        return np.array(eprice_ahead)

    def _scale_action(self, action):
        # if DRL action is in [-1,1]
        action = self.action_space_low + (action + 1.0) * 0.5 * \
            (self.action_space_high - self.action_space_low)
        action = np.clip(action, self.action_space_low, self.action_space_high)
        return action

    
def get_data_from_csv(path, start_time=None, end_time=None):
    """Get a DataFrame of all price variables from csv files
        Parameters
        ----------
        start_time : datetime
            Start time of timeseries
        end_time : datetime
            End time of timeseries
        -------
        df: pandas DataFrame
            DataFrame where each column is a variable in the variables section in the configuration
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    idx = df.loc[start_time:end_time].index
    df = df.loc[idx,]

    return df






