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
        chiller_COP=None,
        boiler_COP=None,
        sim_year=None,
        tz_name=None,
        sim_days=None,
        step_size=None):

        ###
        self.envelope_path = envelope_path
        self.battery_path = battery_path
        self.pv_path = pv_path

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

        # electricity price
        self.eprice_path = eprice_path
        self.eprice_df = pd.read_csv(self.eprice_path, index_col=0, parse_dates=True)

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
        self.envelope_action_names = ["SaFrB", "SaTempB", "LigB_input"]
        self.battery_action_names = ["P_ctrl"]
        # self.pv_action_names = ['scale']
        self.action_names = (
            self.envelope_action_names 
            + self.battery_action_names)
        self.action_init = np.array([0.5, 22.0, 1.0, 0.0])

        # Define Action Space:

        self.action_space = spaces.Box(
            low=np.array([0.1, 10.0, 0.0, -3300.0]),
            high=np.array([2.0, 30.0, 1.0, 3300.0]),
            dtype=np.float32,
        )

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
        # self.envelope_obs_names = ['OutTemp','OutRH','OutSI', \
        #                            'ZoneTempA','LigA','MelsA','FanPowerA','CoolA','HeatA','SolarA', \
        #                            'actual_SaFrA','actual_SaTempA','actual_CwTempA','actual_HwTempA',]
        # self.envelope_obs_names = ['OutTemp','OutRH','OutSI', \
        #                            'ZoneTempA','LigA','MelsA','FanPowerA','CoolA','HeatA','SolarA']
        self.envelope_obs_names = [
            "OutTemp",
            "OutRH",
            "OutSI",
            "ZoneTempB",
            "LigB",
            "FanPowerB",
            "CoolB",
            "HeatB",
            "MelsB",
            "ZoneDayLigIllB"]
        self.battery_obs_names = ["SOC", "P_batt"]
        self.pv_obs_names_1 = ["P1"]
        self.pv_obs_names_2 = ["P2"]
        self.obs_names_simulation = (
            self.envelope_obs_names
            + self.battery_obs_names
            + self.pv_obs_names_1
            + self.pv_obs_names_2)

        self.observation_space_simulation = spaces.Box(
            low=np.array(
                [
                    -20.0,
                    0.0,
                    0.0,
                    5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -4000.0,
                    0.0,
                    0.0,
                    0.0
                ]),
            high=np.array(
                [
                    40.0,
                    100.0,
                    20000.0,
                    35.0,
                    10000.0,
                    10000.0,
                    2000000.0,
                    2000000.0,
                    10000.0,
                    1.0,
                    4000.0,
                    10000.0,
                    10000.0,
                    10000.0
                ]),
            dtype=np.float32)
        # self.obs_names = ['OutTemp','OutRH','OutSI', \
        #                   'ZoneTempA','LigA','MelsA','FanPowerA','CoolA','HeatA','SolarA', \
        #                   'actual_SaFrA','actual_SaTempA','actual_CwTempA','actual_HwTempA',
        #                   'SOC','P_batt','P']
        # self.observation_space = spaces.Box(low =   np.array([-20.0,    0.0,     0.0,  5.0,     0.0,     0.0,     0.0,     0.0,         0.0,       0.0, 0.1, 10.0,  5.0, 30.0, 0.0, -4000.0,     0.0]),
        #                                     high =  np.array([  40.0, 100.0, 20000.0, 35.0, 10000.0, 10000.0, 10000.0, 2000000.0, 2000000.0, 1000000.0, 2.0, 30.0, 15.0, 60.0, 1.0,  4000.0, 10000.0]),
        #                                     dtype = np.float32)
        # self.observation_space = spaces.Box(low =   np.array([-20.0,    0.0,     0.0,  5.0,     0.0,     0.0,     0.0,     0.0,         0.0,       0.0, 0.0, -4000.0,     0.0]),
        #                                     high =  np.array([  40.0, 100.0, 20000.0, 35.0, 10000.0, 10000.0, 10000.0, 2000000.0, 2000000.0, 1000000.0, 1.0,  4000.0, 10000.0]),
        #                                     dtype = np.float32)

        #  observation used by the DRL model:
        self.obs_names = [
            "TOW",
            "OutTemp",
            "OutRH",
            "OutSI",
            "ZoneTempB",
            "net_power",
            "net_energy",
            "ZoneTotalIlluB",
            "SOC",
            "P1",
            "P2"]
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,
                    -20.0, 
                    0.0, 
                    0.0, 
                    5.0, 
                    -5.0, 
                    -1.25, 
                    0.0, 
                    0.0, 
                    0.0, 
                    0.0
                ]),
            high=np.array(
                [
                    167.5,
                    40.0,
                    100.0,
                    20000.0,
                    35.0,
                    20.0,
                    5.0,
                    3000.0,
                    1.0,
                    10000.0,
                    10000.0
                ]),
            dtype=np.float32)

    # def reset(self):
    #     self.time_step_idx = 0
    #     self.episode_idx =+ 1
    #     self.reward = 0.0
    #     self.e_cost = 0.0
    #     self.last_SOC = 0.1
    #     self.total_e_cost = []

    #     if self.episode_idx > -1:

    #         # # unload the FMU by deallocating the resources associated to it from the memory
    #         # self.envelope_model.unload_fmu()
    #         # self.battery_model.unload_fmu()
    #         # self.pv_model.unload_fmu()

    #         print("############## Terminating ##########################")
    #         print("############## Terminating ##########################")

    #         self.envelope_model.terminate_fmu()
    #         self.battery_model.terminate_fmu()
    #         self.pv_model.terminate_fmu()

    #         # load fmu models
    #         self.envelope_model = FmuModel(self.envelope_path)
    #         self.battery_model = FmuModel(self.battery_path)
    #         self.pv_model = FmuModel(self.pv_path)

    #         # set up battery and pv parameters
    #         battery_parameters, pv_parameters, eplus_parameters = load_parameters()
    #         self.battery_model.set_parameters(battery_parameters)
    #         self.pv_model.set_parameters(pv_parameters)
    #         self.envelope_model.set_parameters(eplus_parameters)

    #     self.envelope_model.create_fmu(self.start_time, self.final_time)
    #     self.battery_model.create_fmu(self.start_time, self.final_time)
    #     self.pv_model.create_fmu(self.start_time, self.final_time)
    #     return self.step(self.action_init)[0]

    def reset(self):
        self.time_step_idx = 0
        self.episode_idx =+ 1
        self.reward = 0.0
        self.e_cost = 0.0
        self.last_SOC = 0.1
        self.total_e_cost = []

        if self.episode_idx > -1:

            # unload the FMU by deallocating the resources associated to it from the memory
            self.envelope_model.unload_fmu()
            self.battery_model.unload_fmu()
            self.pv_model_1.unload_fmu()
            self.pv_model_2.unload_fmu()

            # load fmu models
            # self.envelope_model = FmuModel(self.envelope_path)
            # self.battery_model = FmuModel(self.battery_path)
            # self.pv_model = FmuModel(self.pv_path)

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

            net_power = self._compute_net_power(obs_simulation) / 1000.0

            # print("net_power")
            # print(net_power)

            net_energy = (self.step_size / 3600.0) * net_power

            # print("net_energy")
            # print(net_energy)

            total_illu = self._total_illuminance(obs_simulation, action)

            obs = [
                self.tow[self.time_step_idx],
                obs_simulation[self.obs_names_simulation.index("OutTemp")],
                obs_simulation[self.obs_names_simulation.index("OutRH")],
                obs_simulation[self.obs_names_simulation.index("OutSI")],
                obs_simulation[self.obs_names_simulation.index("ZoneTempB")],
                net_power,
                net_energy,
                total_illu,
                obs_simulation[self.obs_names_simulation.index("SOC")],
                obs_simulation[self.obs_names_simulation.index("P1")],
                obs_simulation[self.obs_names_simulation.index("P2")]]

            reward = self._compute_reward(action, net_power, net_energy, obs)

            self.last_SOC = obs[self.obs_names.index("SOC")]

            if self.time_step_idx < (self.n_steps - 1):
                done = False
            else:
                done = True

            self.time_step_idx += 1

            obs = (2.0 * (obs - self.observation_space.low) / \
                (self.observation_space.high - self.observation_space.low) - 1)

        return obs, reward, done, [self.e_cost]# [self.e_cost, net_energy, net_power] #, {}

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

        new_action = np.append(action[0 : len(self.envelope_action_names)],[6.0,60.0])

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

        # print("envelope_obs")
        # print(envelope_obs)
        # print("battery_obs")
        # print(battery_obs)
        # print("pv_obs_1")
        # print(pv_obs_1)
        # print("pv_obs_2")
        # print(pv_obs_2)
        obs = (envelope_obs, battery_obs, pv_obs_1, pv_obs_2)
        obs = np.concatenate(obs)

        return obs

    def _compute_reward(
        self,
        action,
        net_power,
        net_energy,
        obs,
        T_min = [21.0, 15.0],
        T_max = [24.0, 29.0],
        L_setpoint = [500.0, 50.0],
        dc_peak_hour = [16, 21],
        dc_Peak_price = 4500,
        dc_nonPeak_price = 2000,
        alpha = 10.0,
        beta = 15.0,
        gamma = 0.1,
        delta = 0.2):

        time_t = self.time_interval[self.time_step_idx]
        print("#################time#############")
        print(time_t)

        ## demand charge calculation
        # update the peak demand at the begining of new month
        if (time_t.day == 1) & (time_t.hour == 0) & (time_t.minute == 0):
            self._peak_peakHour = 0
            self._peak_partialPeakHour = 0
            self._peak_nonPeakHour = 0

        # only consider the peak hour and non-peak hour
        # if dc_peak_hour[0] <= time_t.hour < dc_peak_hour[1]:
        #     dc_cost = max(0, net_power - self._peak_peakHour) * dc_Peak_price
        #     self._peak_peakHour = max(net_power, self._peak_peakHour)
        # else:
        #     dc_cost = max(0, net_power - self._peak_nonPeakHour) * dc_nonPeak_price
        #     self._peak_nonPeakHour = max(net_power, self._peak_nonPeakHour)
        
        # if dc_cost > 0.0:
        #     print('demand charge at {0}: {1}'.format(time_t, dc_cost))
        #     print("self._peak_peakHour")
        #     print(self._peak_peakHour)
        #     print("self._peak_nonPeakHour")
        #     print(self._peak_nonPeakHour)

        df_price = get_price_data_from_df(
            df=self.eprice_df, 
            start_time=time_t, 
            end_time=time_t)

        if 7 <= time_t.hour < 20:
            T_min_t = T_min[0]
            T_max_t = T_max[0]
            

        else:
            T_min_t = T_min[1]
            T_max_t = T_max[1]

        Zone_DayLight = obs[self.obs_names.index("ZoneTotalIlluB")] - \
            538 * action[self.action_names.index("LigB_input")]

        if 6 <= time_t.hour < 22:
            L_setpoint_t = L_setpoint[0]  # office hour, should maintain 500 lux
            
            if Zone_DayLight >= L_setpoint_t:
                L_comfort_t = abs(538 * action[self.action_names.index("LigB_input")] - 0.0)
            else:
                L_comfort_t = (abs(L_setpoint_t - obs[self.obs_names.index("ZoneTotalIlluB")]))

        else:
            L_setpoint_t = L_setpoint[1]  # non-office hour, should maintain 0 lux
            L_comfort_t = (max(0, (obs[self.obs_names.index("ZoneTotalIlluB")] - \
                L_setpoint_t)) )

            
        # if 6 <= time_t.hour < 22:
        #     L_setpoint_t = L_setpoint[0]  # office hour, should maintain 500 lux
        #     L_comfort_t = (abs(L_setpoint_t - obs[self.obs_names.index("ZoneTotalIlluB")]))

        # else:
        #     L_setpoint_t = L_setpoint[1]  # non-office hour, should maintain 0 lux
        #     L_comfort_t = (max(0, (obs[self.obs_names.index("ZoneTotalIlluB")] - \
        #         L_setpoint_t)) )

        # if 6 <= time_t.hour < 22:
        #     L_setpoint_t = L_setpoint[0]  # office hour, should maintain 500 lux
        #     L_comfort_t = (max(0, (L_setpoint_t - \
        #         obs[self.obs_names.index("ZoneTotalIlluB")])) )

        # else:
        #     L_setpoint_t = L_setpoint[1]  # non-office hour, should maintain 0 lux
        #     L_comfort_t = (max(0, (obs[self.obs_names.index("ZoneTotalIlluB")] - \
        #         L_setpoint_t)) )
        #     # if action[self.action_names.index("LigB_input")] > 0.0:
        #     #     L_comfort_t = 100.0/gamma
        #     # else:
        #     #     L_comfort_t = 0.0

        e_price = df_price["e_price"].iloc[-1]  

        if net_energy < 0.0:
            net_energy = 0.0

        e_cost_t = 100*e_price * net_energy #+ dc_cost  # e_price/100.0 * net_energy

        T_comfort_t = max(0, (obs[self.obs_names.index("ZoneTempB")] - T_max_t)) + \
             max(0, (T_min_t - obs[self.obs_names.index("ZoneTempB")]))

        reward_t = -1.0 * (alpha * e_cost_t + beta * T_comfort_t + gamma * L_comfort_t)

        # print("ZoneDayLigIllB")
        # print(obs[self.obs_names.index("ZoneTotalIlluB")] - 538 * action[self.action_names.index("LigB_input")])


        # print("LED")
        # print(538 * action[self.action_names.index("LigB_input")])

        print("LigB_input")
        print(action[self.action_names.index("LigB_input")])

        # print("e_cost_t")
        # print(alpha*e_cost_t)
        # print("T_comfort_t")
        # print(beta * T_comfort_t)
        # print("L_comfort_t")
        # print(gamma*L_comfort_t)
        # print("net_energy")
        # print(net_energy)
        # print("e_price")
        # print(e_price)

        net_kWh_battery = (1.0 / 1000.0 * (self.step_size / 3600.0) * \
             action[self.action_names.index("P_ctrl")])
        # print("net_kWh_battery")
        # print(net_kWh_battery)
        SOC_kWh_min = self.last_SOC * 7.0 - 0.7
        SOC_kWh = self.last_SOC * 7.0
        # print("SOC_kWh_min")
        # print(SOC_kWh_min)
        # print("SOC_kWh")
        # print(SOC_kWh)
        # print("reward_t_before")
        # print(reward_t)
        if net_kWh_battery < 0.0 and abs(net_kWh_battery) > SOC_kWh_min:
            reward_t = reward_t - 200.0

        elif net_kWh_battery > 0.0 and (net_kWh_battery + SOC_kWh) > 7.0:
            reward_t = reward_t - 200.0

        # print("reward_t_after")
        # print(reward_t)

        self.reward = reward_t
        self.e_cost = e_cost_t
        self.total_e_cost.append(e_cost_t)
        return self.reward

    def _compute_net_power(self, obs):
        net_power = (
            obs[self.obs_names_simulation.index("LigB")]
            + obs[self.obs_names_simulation.index("FanPowerB")]
            + obs[self.obs_names_simulation.index("CoolB")] / self.chiler_cop
            + obs[self.obs_names_simulation.index("HeatB")] / self.boiler_cop
            + obs[self.obs_names_simulation.index("MelsB")]
            + obs[self.obs_names_simulation.index("P_batt")]
            - obs[self.obs_names_simulation.index("P1")]
            - obs[self.obs_names_simulation.index("P2")])

        print("LigB")
        print(obs[self.obs_names_simulation.index("LigB")])
        return net_power

    def _total_illuminance(self, obs, action):
        total_illuminance = (
            obs[self.obs_names_simulation.index("ZoneDayLigIllB")]
            + action[self.action_names.index("LigB_input")] * 538)

        # print("ZoneDayLigIllB")
        # print(obs[self.obs_names_simulation.index("ZoneDayLigIllB")])
        
        
        # WHAT IS 538??????
        # print('Daylighting: {0} lux; Total Lighting: {1} lux'.format(
        #    obs[self.obs_names_simulation.index("ZoneDayLigIllB")], total_illuminance))
        return total_illuminance


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


def get_price_data_from_csv(eprice_path, start_time=None, end_time=None):
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
    df = pd.read_csv(eprice_path, index_col=0, parse_dates=True)
    # df = df.tz_localize(tz_local).tz_convert(tz_utc)
    df.index = pd.to_datetime(df.index)
    idx = df.loc[start_time:end_time].index
    df = df.loc[idx,]

    return df


def get_price_data_from_df(df, start_time=None, end_time=None):
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

    def append(self, action, obs, reward, e_cost):  # , net_energy, net_power):
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

    # def net_energy_data(self):
    #     return self.net_energy_df

    # def net_power_data(self):
    #     return self.net_power_df


# TODO: debug the class some error occurs at the FMU environment level
# class EnvsBatch:
#     def __init__(self,
#                  envelope_dir,
#                  battery_path,
#                  pv_dir,
#                  eprice_path,
#                  chiller_COP = 3.0,
#                  boiler_COP = 0.95,
#                  tz_name = 'America/Los_Angeles',
#                  sim_days = 365,
#                  step_size = 900):
#         # initialize model observation
#         self.envelope_dir = envelope_dir
#         self.envelope_path_list = os.listdir(envelope_dir)
#         self.battery_path = battery_path
#         self.pv_dir = pv_dir
#         self.eprice_path = eprice_path
#         self.chiller_COP = chiller_COP
#         self.boiler_COP = boiler_COP
#         self.tz_name = tz_name
#         self.sim_days = sim_days
#         self.step_size = step_size

#     def sample_env(self):
#         idx = randint(0,(len(self.envelope_path_list)-1))
#         print("idx")
#         print(idx)
#         envelope_file_name = self.envelope_path_list[idx]
#         envelope_path = os.path.join(self.envelope_dir, envelope_file_name)
#         print("envelope_path")
#         print(envelope_path)
#         envelope_file_name = envelope_file_name[:-len(".fmu")]
#         envelope_file_name = envelope_file_name.split("_")[-2:]
#         pv_file_name = "PV"+"_"+ envelope_file_name[0]+ "_"+ envelope_file_name[1]+".fmu"
#         pv_path = os.path.join(self.pv_dir, pv_file_name)
#         print("pv_path")
#         print(pv_path)
#         if envelope_file_name[1]=="TMY":
#             sim_year = 2017
#         else:
#             sim_year = int(envelope_file_name[1])

#         print("sim_year")
#         print(sim_year)

#         self.env = FlexLabEnv(envelope_path = envelope_path,
#                          battery_path = self.battery_path,
#                          pv_path =  pv_path,
#                          eprice_path = self.eprice_path,
#                          chiller_COP = self.chiller_COP,
#                          boiler_COP = self.boiler_COP,
#                          sim_year = sim_year,
#                          tz_name = self.tz_name,
#                          sim_days = self.sim_days,
#                          step_size = self.step_size)

#         return self.env

#     def stop_env(self):
#         self.env.stop_fmu
#         os.remove("battery_log.txt")
#         os.remove("PV_log.txt")
#         os.remove("FlexlabXR_v1_Eco_log.txt")
#         #del self.env
