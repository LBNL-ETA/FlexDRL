# -*- coding: utf-8 -*-
"""

"""

import os
from datetime import timedelta
import pandas as pd
import numpy as np
import math
import pytz
from pyfmi import load_fmu


from models.fmu_models import FmuModel

from random import randint

from gym import Env
from gym import spaces
from gym.utils import seeding


class FlexLabEnv(Env):
    """ FlexLabEnv is a custom Gym Environment 
    Args:

    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        envelope_path=None,
        battery_path=None,
        pv_path=None,
        eprice_path=None,
        daylight_path=None,
        chiller_COP=None,
        boiler_COP=None,
        sim_year=None,
        tz_name=None,
        sim_days=None,
        step_size=None,
        eprice_ahead = None,
        alpha_r=None,
        beta_r=None,
        gamma_r=None,
        delta_r=None,
        pv_panels=None,
        light_ctrl = False):

        ###
        self.envelope_path = envelope_path
        self.battery_path = battery_path
        self.pv_path = pv_path

        # reward function parameters
        self.alpha_r = alpha_r
        self.beta_r = beta_r
        self.gamma_r = gamma_r
        self.delta_r = delta_r

        #
        self.eprice_ahead = eprice_ahead

        #
        self.pv_panels = pv_panels

        #
        self.light_ctrl = light_ctrl


        ###
        self.chiler_cop = chiller_COP
        self.boiler_cop = boiler_COP

        # define simulations time steps
        self.tz_local = pytz.timezone(tz_name)
        self.tz_utc = pytz.timezone("UTC")
        self.start_time = 0
        self.final_time = 3600 * 24 * sim_days
        self.step_size = step_size  # 900 == 15 min
        self.time_steps = np.arange(self.start_time, self.final_time, self.step_size)
        self.n_steps = len(self.time_steps)
        self.sim_year = sim_year
        self.time_interval = pd.date_range(
            start="1/1/{0}".format(sim_year),
            end="1/1/{0}".format(sim_year + 1),
            periods=35040 + 1)  # 35040 is for 15 min interval
        self.tow = self.time_interval.hour + \
             self.time_interval.minute/60.0 + \
                  self.time_interval.weekday * 24
        self.tod = self.time_interval.hour + \
             self.time_interval.minute/60.0 

        # electricity price
        self.eprice_path = eprice_path
        self.eprice_df = pd.read_csv(self.eprice_path, index_col=0, parse_dates=True)

        # daylight
        self.daylight_path = daylight_path
        self.daylight_df = pd.read_csv(self.daylight_path, index_col=0, parse_dates=True)

        # store peak demand to calculate demand charge
        self._peak_peakHour = 0
        self._peak_partialPeakHour = 0
        self._peak_nonPeakHour = 0

        # initiate np array to store the results
        self.heat = np.empty(self.n_steps + 1)  # control signal
        self.shade = np.empty(self.n_steps + 1)  # control signal
        self.room_temp = np.empty(self.n_steps)
        self.lig = np.empty(self.n_steps)
        self.mels = np.empty(self.n_steps)
        self.outdoor_temp = np.empty(self.n_steps)
        self.outdoor_rh = np.empty(self.n_steps)
        self.heat_input = np.empty(self.n_steps)
        self.solar = np.empty(self.n_steps)

        # other initializations

        self.episode_idx = -1

        # # load fmu models

        self.envelope_model = FmuModel(envelope_path)
        self.battery_model = FmuModel(battery_path)
        self.pv_model_1 = FmuModel(pv_path)
        self.pv_model_2 = FmuModel(pv_path)

        # set up battery and pv parameters
        (battery_parameters, 
        pv_parameters_1, 
        pv_parameters_2, 
        eplus_parameters) = load_parameters()
        self.battery_model.set_parameters(battery_parameters)
        self.pv_model_1.set_parameters(pv_parameters_1)
        self.pv_model_2.set_parameters(pv_parameters_2)
        self.envelope_model.set_parameters(eplus_parameters)

        # Initialize action
        """
        Envelope:
            SaFr: AHU supply air (to room) flow rate, [kg/s]
            SaTemp: AHU supply air (to room) temperature, [degC]
            CwTemp: Chiller supply water (to AHU) temperature, [degC]
            HwTemp: Boiler supply water (to AHU) temperature, [degC]
            Shade: shading control signal, binary, 0 for shading off (solar irradiation unblocked); 1 for shading on
            Lig_input: lighting control signal, continous between 0 and 1, 1 for 8.26W/m2 lighting energy, 0 for lighting off
        Battery:
            P_ctrl: Power control to charge(positive) discharge(negative) the battery [W]
	        for the battery in Flexlab, should be in the range of -3300 ~ 3300
        PV:
            scale: Shading of PV module [1], 1 - no shading, full generation capacity
        """

        self.battery_action_names = ["P_ctrl"]

        if self.light_ctrl:
            self.envelope_action_names = ["SaFrB", "SaTempB", "LigB_input"]
            self.action_init = np.array([0.5, 22.0, 1.0, 0.0])
            self.action_space = spaces.Box(
                low=np.array([0.05, 10.0, 0.0, -3300.0]),
                high=np.array([2.0, 30.0, 1.0, 3300.0]),
                dtype=np.float32)
            
        else:
            self.envelope_action_names = ["SaFrB", "SaTempB"]
            self.action_init = np.array([0.5, 22.0, 0.0])
            self.action_space = spaces.Box(
                low=np.array([0.05, 10.0, -3300.0]),
                high=np.array([2.0, 30.0, 3300.0]),
                dtype=np.float32)

        self.action_names = (
            self.envelope_action_names 
            + self.battery_action_names)

        #  Define Observation Space
        """
        Envelope:
            OutTemp,OutRH,OutSI: Outdoor temperature, relative humidity, and solar irradiantion
            ZoneTemp: Indoor air temperature
            Lig,Mels,FanPower: electricity power of lighting, plug load, and supply air fan, unit W
            Cool,Heat: cooling, heating load 
            Solar: load from solar irradiation
            actual_SaFr,actual_SaTemp,actual_CwTemp,actual_HwTemp: implemented value of inputs, to check whether the inputs are correctly implemented
        Battery:
            SOC: State of Charge [1]
            P_batt: Power actually consumed by the battery: charge(positive) discharge(negative)
        PV:
            P: Active power [W]
        """

        self.envelope_obs_names = [
            "OutTemp",
            "OutRH",
            "OutSI",
            "ZoneTempB",
            "LigB",
            "FanPowerB",
            "CoolB",
            "HeatB",
            "MelsB"]
        self.battery_obs_names = ["SOC", "P_batt"]
        self.pv_obs_names_1 = ["P1"]
        self.pv_obs_names_2 = ["P2"]
        self.obs_names_simulation = (
            self.envelope_obs_names
            + self.battery_obs_names
            + self.pv_obs_names_1
            + self.pv_obs_names_2)

        #  observation used by the DRL model:

        if self.light_ctrl:

            main_obs_names = [
                "TOD", "OutTemp", "OutRH", "OutSI",
                "ZoneTempB", "net_power","net_energy",
                "ZoneTotalIlluB", "SOC", "PV"]
            main_obs_low = np.array([
                0.0, -20.0, 0.0, 0.0, 
                5.0, -5.0, -1.25, 
                0.0, 0.0, 0.0])
            main_obs_high = np.array([
                23.75, 40.0, 100.0, 20000.0,
                35.0, 20.0, 5.0,
                1200.0, 1.0, 5000.0])

        else:

            main_obs_names = [
                "TOD", "OutTemp", "OutRH", "OutSI",
                "ZoneTempB", "net_power","net_energy",
                "SOC", "PV"]
            main_obs_low = np.array([
                0.0, -20.0, 0.0, 0.0, 
                5.0, -5.0, -1.25, 
                0.0, 0.0])
            main_obs_high = np.array([
                23.75, 40.0, 100.0, 20000.0,
                35.0, 20.0, 5.0,
                1.0, 5000.0])

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

        obs_names = main_obs_names + eprice_x_ahead

        self.obs_names = obs_names

        min_eprice = min(self.eprice_df['e_price'])
        max_eprice = max(self.eprice_df['e_price'])

        eprice_x_ahead_low = np.repeat(min_eprice, eprice_ahead)
        eprice_x_ahead_high = np.repeat(max_eprice, eprice_ahead)

        obs_space_low = (main_obs_low,eprice_x_ahead_low)
        obs_space_low = np.concatenate(obs_space_low)
        obs_space_high = (main_obs_high,eprice_x_ahead_high)
        obs_space_high = np.concatenate(obs_space_high)

        self.observation_space = spaces.Box(
            low=obs_space_low,
            high=obs_space_high,
            dtype=np.float32)

       

    def reset(self):
        self.time_step_idx = 0
        self.episode_idx =+ 1
        self.reward = 0.0
        self.e_cost = 0.0
        self.last_SOC = 0.1
        self.total_e_cost = []
        self.total_e_cost_reward = 0.0
        self.total_T_reward = 0.0
        if self.light_ctrl:
            self.total_L_reward = 0.0

        if self.episode_idx > -1:

            # unload the FMU by deallocating the resources associated to it from the memory
            self.envelope_model.unload_fmu()
            self.battery_model.unload_fmu()
            self.pv_model_1.unload_fmu()
            self.pv_model_2.unload_fmu()

            # set up battery and pv parameters
            (
                battery_parameters,
                pv_parameters_1,
                pv_parameters_2,
                eplus_parameters,
            ) = load_parameters()
            self.battery_model.set_parameters(battery_parameters)
            self.pv_model_1.set_parameters(pv_parameters_1)
            self.pv_model_2.set_parameters(pv_parameters_2)
            self.envelope_model.set_parameters(eplus_parameters)

        self.envelope_model.create_fmu(self.start_time, self.final_time)
        self.battery_model.create_fmu(self.start_time, self.final_time)
        self.pv_model_1.create_fmu(self.start_time, self.final_time)
        self.pv_model_2.create_fmu(self.start_time, self.final_time)
        return self.step(self.action_init)[0]

    def stop_fmu(self):
        self.envelope_model.terminate_fmu()
        self.battery_model.terminate_fmu()
        self.pv_model_1.terminate_fmu()
        self.pv_model_2.terminate_fmu()

    def step(self, action):
        if action is not None:

            # rescale to the original scale

            action = self.scale_action(action)

            obs_simulation = self._take_action(action)

            net_power = self._compute_net_power(obs_simulation,action) / 1000.0

            net_energy = (self.step_size / 3600.0) * net_power

            PV_power = self.pv_panels/14.0*(
                obs_simulation[self.obs_names_simulation.index("P1")] 
                + obs_simulation[self.obs_names_simulation.index("P2")])

            if self.light_ctrl:
                total_illu = self._total_illuminance(obs_simulation, action)
                obs = [
                self.tod[self.time_step_idx],
                    obs_simulation[self.obs_names_simulation.index("OutTemp")],
                    obs_simulation[self.obs_names_simulation.index("OutRH")],
                    obs_simulation[self.obs_names_simulation.index("OutSI")],
                    obs_simulation[self.obs_names_simulation.index("ZoneTempB")],
                    net_power,
                    net_energy,
                    total_illu,
                    obs_simulation[self.obs_names_simulation.index("SOC")],
                    PV_power]


            else:
                obs = [
                self.tod[self.time_step_idx],
                    obs_simulation[self.obs_names_simulation.index("OutTemp")],
                    obs_simulation[self.obs_names_simulation.index("OutRH")],
                    obs_simulation[self.obs_names_simulation.index("OutSI")],
                    obs_simulation[self.obs_names_simulation.index("ZoneTempB")],
                    net_power,
                    net_energy,
                    obs_simulation[self.obs_names_simulation.index("SOC")],
                    PV_power]

            
            if self.eprice_ahead != 0:
                eprices_ahead = self._eprice_ahead(self.eprice_ahead)
                obs = (obs, eprices_ahead)
                obs = np.concatenate(obs)

            reward = self._compute_reward(action, net_power, net_energy, obs)

            self.last_SOC = obs[self.obs_names.index("SOC")]

            if self.time_step_idx < (self.n_steps - 1):
                done = False
            else:
                done = True
                print("total_e_cost_reward")
                print(self.total_e_cost_reward)
                print("total_T_reward")
                print(self.total_T_reward)
                if self.light_ctrl:
                    print("total_L_reward")
                    print(self.total_L_reward)

            self.time_step_idx += 1

            obs = (2.0 * (obs - self.observation_space.low) / \
                (self.observation_space.high - self.observation_space.low) - 1)

        safrB = action[self.action_names.index("SaFrB")]
        self.fan_power = 1494.7*safrB**3 + 307.25*safrB**2 - 7.6768*safrB + 54.018
        # self.fan_power = obs_simulation[self.obs_names_simulation.index("FanPowerB")]
        self.cool_power = obs_simulation[self.obs_names_simulation.index("CoolB")] / self.chiler_cop
        self.heat_power = obs_simulation[self.obs_names_simulation.index("HeatB")] / self.boiler_cop
        self.plugs_load = obs_simulation[self.obs_names_simulation.index("MelsB")]

        return obs, reward, done, [self.e_cost,self.fan_power,self.cool_power,self.heat_power,self.plugs_load] #, {}

    def scale_obs(self, obs):

        # if DRL obs is in [-1,1]
        obs_scaled = self.observation_space.low + (obs + 1.0) * 0.5 * \
            (self.observation_space.high - self.observation_space.low)
        obs_scaled = np.clip(
            obs_scaled, 
            self.observation_space.low, 
            self.observation_space.high)

        return obs_scaled

    def scale_action(self, action):
        # if DRL action is in [-1,1]
        action = self.action_space.low + (action + 1.0) * 0.5 * \
            (self.action_space.high - self.action_space.low)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def _take_action(self, action):

        new_envelope_action_names = ["SaFrB", "SaTempB", "LigB_input", "CwTempB", "HwTempB"]

        if self.light_ctrl:
            new_action = np.append(action[0 : len(self.envelope_action_names)],[6.0,60.0])

        else:
            time_t = self.time_interval[self.time_step_idx]

            if 7 <= time_t.hour < 20:
                new_action = np.append(action[0 : len(self.envelope_action_names)],[1.0,6.0,60.0])
            else:
                new_action = np.append(action[0 : len(self.envelope_action_names)],[0.0,6.0,60.0])

        envelope_obs = self.envelope_model.simulate(
            new_envelope_action_names,
            new_action,
            self.envelope_obs_names,
            self.time_steps,
            self.step_size,
            self.time_step_idx)

        battery_obs = self.battery_model.simulate(
            self.battery_action_names,
            [action[(len(self.action_names) - 1)]],
            self.battery_obs_names,
            self.time_steps,
            self.step_size,
            self.time_step_idx)

        pv_obs_1 = self.pv_model_1.simulate(
            ["scale"], [1.0], ["P"], 
            self.time_steps, 
            self.step_size, 
            self.time_step_idx)

        pv_obs_2 = self.pv_model_2.simulate(
            ["scale"], [1.0], ["P"], 
            self.time_steps, 
            self.step_size, 
            self.time_step_idx)

        obs = (envelope_obs, battery_obs, pv_obs_1, pv_obs_2)
        obs = np.concatenate(obs)

        return obs

    def _compute_net_power(self, obs, action):
        PV_power = self.pv_panels/14.0*(
            obs[self.obs_names_simulation.index("P1")]
            + obs[self.obs_names_simulation.index("P2")])

        # update FanPowerB based on Flexlab Data
        safrB = action[self.action_names.index("SaFrB")]
        fan_power = 1494.7*safrB**3 + 307.25*safrB**2 - 7.6768*safrB + 54.018

        # plugs load emulation
        time_t = self.time_interval[self.time_step_idx]
        if 8 <= time_t.hour < 18:
            plugs_load = 1400.0
        else:
            plugs_load = 50.0

        net_power = (
            obs[self.obs_names_simulation.index("LigB")]
            + fan_power
            + obs[self.obs_names_simulation.index("CoolB")] / self.chiler_cop
            + obs[self.obs_names_simulation.index("HeatB")] / self.boiler_cop
            + plugs_load
            + obs[self.obs_names_simulation.index("P_batt")]
            - PV_power)
        
        # obs[self.obs_names_simulation.index("FanPowerB")]

        return net_power

    def _total_illuminance(self, obs, action):
        time_t = self.time_interval[self.time_step_idx]
        daylight = get_data_from_df(
            df=self.daylight_df, 
            start_time=time_t, 
            end_time=time_t)
        daylight_t = daylight["Daylighting_B"].iloc[-1]  
        total_illuminance = (
            daylight_t
            + action[self.action_names.index("LigB_input")] * 538)

        return total_illuminance

    def _eprice_ahead(self, hours_ahead):
        times_ahead = (hours_ahead - 1) * 4
        if self.time_step_idx > (35040 - times_ahead - 1):
            start_t_idx = self.time_step_idx - (times_ahead + 1)
        else:
            start_t_idx = self.time_step_idx
        start_t = self.time_interval[start_t_idx]
        end_t = self.time_interval[start_t_idx + times_ahead]
        df = get_data_from_df(
            df=self.eprice_df, 
            start_time=start_t, 
            end_time=end_t)
        eprice_ahead = []
        for i in range(0,df.shape[0],4):
            eprice_ahead.append(df.iloc[i]["e_price"])
        return np.array(eprice_ahead)


    def _compute_reward(
        self,
        action,
        net_power,
        net_energy,
        obs,
        T_min = [21.0, 15.0],
        T_max = [24.0, 29.0],
        L_setpoint = [400.0, 50.0]):

        reward = self._compute_default_reward(action, net_power, net_energy, obs, T_min, T_max, L_setpoint)

        return reward

    # Define Reward Function

    def _compute_default_reward(
        self,
        action,
        net_power,
        net_energy,
        obs,
        T_min = [21.0, 15.0],
        T_max = [24.0, 29.0],
        L_setpoint = [400.0, 50.0]):

        time_t = self.time_interval[self.time_step_idx]

        df_price = get_data_from_df(
            df=self.eprice_df, 
            start_time=time_t, 
            end_time=time_t)

        e_price = df_price["e_price"].iloc[-1] 

        if net_energy <= 0.0:
            e_cost_t = 0.0
        else:
            e_cost_t = e_price * net_energy

        if 7 <= time_t.hour < 19:
            T_min_t = T_min[0]
            T_max_t = T_max[0]
            T_comfort_t = 1 -math.exp(-1.0/2.0*(obs[self.obs_names.index("ZoneTempB")] - 22.5)**2) + \
                0.2*(max(0, (obs[self.obs_names.index("ZoneTempB")] - T_max_t)) + \
                max(0, (T_min_t - obs[self.obs_names.index("ZoneTempB")])))
            
        else:
            T_min_t = T_min[1]
            T_max_t = T_max[1]
            T_comfort_t = 0.2*(max(0, (obs[self.obs_names.index("ZoneTempB")] - T_max_t)) + \
                max(0, (T_min_t - obs[self.obs_names.index("ZoneTempB")])))

        if self.light_ctrl:
            if 7 <= time_t.hour < 20:
                L_setpoint_t = L_setpoint[0]  # office hour, should maintain 500 lux
                L_comfort_t = (max(0, (L_setpoint_t - \
                    obs[self.obs_names.index("ZoneTotalIlluB")])))/538.0

            else:
                L_setpoint_t = L_setpoint[1]  # non-office hour, should maintain 0 lux
                L_comfort_t = (max(0, (obs[self.obs_names.index("ZoneTotalIlluB")] - \
                    L_setpoint_t)))/538.0

            reward_t = -1.0 * (
                self.alpha_r * e_cost_t 
                + self.beta_r * T_comfort_t 
                + self.gamma_r * L_comfort_t)

        else:
             reward_t = -1.0 * (
                self.alpha_r * e_cost_t 
                + self.beta_r * T_comfort_t)


        net_kWh_battery = (1.0 / 1000.0 * (self.step_size / 3600.0) * \
             action[self.action_names.index("P_ctrl")])

        SOC_kWh_min = self.last_SOC * 7.0 - 0.7
        SOC_kWh = self.last_SOC * 7.0

        if net_kWh_battery < 0.0 and abs(net_kWh_battery) > SOC_kWh_min:
            reward_t = reward_t - 20.0

        elif net_kWh_battery > 0.0 and (net_kWh_battery + SOC_kWh) > 7.0:
            reward_t = reward_t - 20.0

        self.reward = reward_t
        self.e_cost = e_cost_t
        self.total_e_cost.append(e_cost_t)
        self.total_e_cost_reward = self.total_e_cost_reward + self.alpha_r * e_cost_t
        self.total_T_reward = self.total_T_reward + self.beta_r * T_comfort_t
        if self.light_ctrl:
            self.total_L_reward = self.total_L_reward + self.gamma_r * L_comfort_t
        return self.reward



def time_converter(year, idx, total_timestep=35040):
    """
    Input
    -------------------
    year: int, year to be converted
    idx: int, index of timestep in the specific year, SHOULD start from zero
    total_timestep: total timestep of the specific year
    
    Output
    -------------------
    pandas Timestamp of the time corresponding to the idx
    """
    index = pd.date_range(
        start="1/1/{0}".format(year),
        end="1/1/{0}".format(year + 1),
        periods=total_timestep + 1)
    time = index[idx]
    return time


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
    # df = df.tz_localize(tz_local).tz_convert(tz_utc)
    df.index = pd.to_datetime(df.index)
    idx = df.loc[start_time:end_time].index
    df = df.loc[idx,]

    return df


def get_data_from_df(df, start_time=None, end_time=None):
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
    df.index = pd.to_datetime(df.index)
    idx = df.loc[start_time:end_time].index
    df = df.loc[idx,]

    return df


def load_parameters():
    # battery model parameters
    """
    EMax: Battery capacity [Wh]
    SOC_start: Initial SOC value [1]
    SOC_min: Minimum SOC value [1]
    SOC_max: Maximum SOC value [1]
    etaCha: Charging efficiency [1]
    etaDis: Discharging efficiency [1]
    """
    battery_parameters = {
        "EMax": 7000,
        "SOC_start": 0.1,
        "SOC_min": 0.1,
        "SOC_max": 1,
        "etaCha": 0.96,
        "etaDis": 0.96}

    # pv model parameters
    """
    weather_file: path to weather file, does not work if the fmu is exported from pymodelica
    n: Number of PV modules [1]
    A: Net surface area per module [m2]
    eta: Module conversion efficiency [1]
    lat: Latitude [deg]
    til: Surface tilt [deg]
    azi: Surface azimuth: 0-S, 90-W, 180-N, 270-E [deg]
    scale: Shading of PV module [1], 1 - no shading, full generation capacity
    """
    pv_parameters_1 = {
        "weather_file": "/home/walter/git/DER/FlexLab_Toy_Model/utilities/ModelicaModel/weather_mos/flexlab_2018.mos",
        "n": 7,
        "A": 1.65,
        "eta": 0.158,
        "lat": 37.9,
        "til": 8,
        "azi": 90}

    pv_parameters_2 = {
        "weather_file": "/home/walter/git/DER/FlexLab_Toy_Model/utilities/ModelicaModel/weather_mos/flexlab_2018.mos",
        "n": 7,
        "A": 1.65,
        "eta": 0.158,
        "lat": 37.9,
        "til": 8,
        "azi": 270}

    # EnergyPlus model parameters
    """
    CwTemp: Chiller supply water (to AHU) temperature, [degC]
    HwTemp: Boiler supply water (to AHU) temperature, [degC]
    Shade: shading control signal, binary, 0 for shading off (solar irradiation unblocked); 1 for shading on
    """
    eplus_parameters = {
        "CwTempB": 6.0, 
        "HwTempB": 60.0, 
        "ShadeB": 0.0}

    return battery_parameters, pv_parameters_1, pv_parameters_2, eplus_parameters


class ExperienceBuffer:
    def __init__(self, obs_names, action_names):
        # initialize model observation
        self.obs_names = obs_names
        self.action_names = action_names

        obs_dict = dict()
        for obs_i in obs_names:
            obs_dict[obs_i] = [0.0]
        self.obs_df = pd.DataFrame(obs_dict)

        # initialize actions
        action_dict = dict()
        for action_i in self.action_names:
            action_dict[action_i] = [0.0]
        self.actions_df = pd.DataFrame(action_dict)

        # initialize rewards
        self.rewards_df = pd.DataFrame({"reward": [0.0]})

        # initialize e_cost
        self.e_costs_df = pd.DataFrame({"e_cost": [0.0]})

        # # initialize net_energy
        # self.net_energy_df = pd.DataFrame({'net_energy':[0.0]})

        # # initialize net_power
        # self.net_power_df = pd.DataFrame({'net_power':[0.0]})

        # initialize fan_power
        self.fan_power_df = pd.DataFrame({"fan_power": [0.0]})

        # initialize cool_power
        self.cool_power_df = pd.DataFrame({"cool_power": [0.0]})

        # initialize heat_power
        self.heat_power_df = pd.DataFrame({"heat_power": [0.0]})

        # initialize plugs_load
        self.plugs_load_df = pd.DataFrame({"plugs_load": [0.0]})

    # [self.e_cost,self.fan_power,self.cool_power,self.heat_power,self.plugs_load] 

    def append(
        self, action, obs, 
        reward, e_cost, fan_power, 
        cool_power, heat_power, plugs_load):

        action_dict = dict()
        for i in range(len(self.action_names)):
            action_i = self.action_names[i]
            action_dict[action_i] = [action[i]]
        action_df_0 = pd.DataFrame(action_dict)
        self.actions_df = self.actions_df.append(action_df_0, ignore_index=True)

        obs_dict = dict()
        for i in range(len(self.obs_names)):
            obs_i = self.obs_names[i]
            obs_dict[obs_i] = [obs[i]]
        obs_df_0 = pd.DataFrame(obs_dict)
        self.obs_df = self.obs_df.append(obs_df_0, ignore_index=True)

        e_costs_df_0 = pd.DataFrame({"e_cost": [e_cost]})
        self.e_costs_df = self.e_costs_df.append(e_costs_df_0, ignore_index=True)

        reward_df_0 = pd.DataFrame({"reward": [reward]})
        self.rewards_df = self.rewards_df.append(reward_df_0, ignore_index=True)

        fan_power_df_0 = pd.DataFrame({"fan_power": [fan_power]})
        self.fan_power_df = self.fan_power_df.append(fan_power_df_0, ignore_index=True)

        cool_power_df_0 = pd.DataFrame({"cool_power": [cool_power]})
        self.cool_power_df = self.cool_power_df.append(cool_power_df_0, ignore_index=True)

        heat_power_df_0 = pd.DataFrame({"heat_power": [heat_power]})
        self.heat_power_df = self.heat_power_df.append(heat_power_df_0, ignore_index=True)

        plugs_load_df_0 = pd.DataFrame({"plugs_load": [plugs_load]})
        self.plugs_load_df = self.plugs_load_df.append(plugs_load_df_0, ignore_index=True)

        # net_energy_df_0 = pd.DataFrame({'net_energy':[net_energy]})
        # self.net_energy_df = self.net_energy_df.append(net_energy_df_0, ignore_index=True)

        # net_power_df_0 = pd.DataFrame({'net_power':[net_power]})
        # self.net_power_df = self.net_power_df.append(net_power_df_0, ignore_index=True)

    def last_action(self):
        return self.actions_df.iloc[len(self.actions_df) - 1]

    def action_data(self):
        return self.actions_df

    def obs_data(self):
        return self.obs_df

    def reward_data(self):
        return self.rewards_df

    def e_cost_data(self):
        return self.e_costs_df

    def fan_power_data(self):
        return self.fan_power_df

    def cool_power_data(self):
        return self.cool_power_df

    def heat_power_data(self):
        return self.heat_power_df

    def plugs_load_data(self):
        return self.plugs_load_df

    # def net_energy_data(self):
    #     return self.net_energy_df

    # def net_power_data(self):
    #     return self.net_power_df
