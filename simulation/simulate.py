# -*- coding: utf-8 -*-
"""
This script runs a simulation of the MPC controller for the convenience
store.  A version of the MPC model with feedback control is used to
represent the real building.

"""

from gym_flexlab.envs import flexlab_env
import os
from datetime import timedelta
import pandas as pd
import numpy as np
import pytz
import random
import matplotlib
#matplotlib.use('Agg')
import pylab as plt
#import matplotlib.pyplot as plt

random.seed(200)

def controller(T_room, Q_pre, T_set=22):
    '''
    A simple controller for hvac input heat
    
    '''    
    if T_room-T_set > 2:
        Q = Q_pre-1000
    elif T_room-T_set >1:
        Q = Q_pre-500
    elif T_room-T_set < -3:
        Q = Q_pre+2000
    elif T_room-T_set < -2:
        Q = Q_pre+1000
    elif T_room-T_set < -1:
        Q = Q_pre+500
    else:
        Q = Q_pre
    return Q

def shade_controller(solar,heat):
    '''
    A simple controller for shading
    
    '''  
    if 0.35*solar + heat > 0:
        shade = 0
    else:
        shade = 1
    return shade

def controller_cooling():
    '''
    A simple HVAC controller outputing the same value
    
    '''    
    airFR = 0.7
    airTemp = 12
    cWaterTemp = 10  
    hWaterTemp = 45
    shade = 1
    lig_input = 0
    P_ctrl = -500
    scale = 1
    return airFR, airTemp, cWaterTemp, hWaterTemp, shade, lig_input, P_ctrl, scale



def test_agent(new_state, action_df):

    # Calculate the input for the next step
    # heat = controller(new_state[0], action_df['QA'], T_set=22)
    # shade = shade_controller(new_state[6],action_df['QA'])
    
    action = controller_cooling()# np.array([heat, shade])

    return np.asarray(action)



if __name__ == '__main__':

    envelope_action_names = ['SaFrA','SaTempA','CwTempA','HwTempA','ShadeA','LigA_input']
    battery_action_names = ['P_ctrl']
    pv_action_names = ['scale']
    action_names = envelope_action_names + battery_action_names + pv_action_names

    obs_names = ['OutTemp','OutRH','OutSI', \
                 'ZoneTempA','LigA','MelsA','FanPowerA','CoolA','HeatA','SolarA', \
	             'actual_SaFrA','actual_SaTempA','actual_CwTempA','actual_HwTempA', \
                 'SOC','P_batt','P'] 


    buffer = flexlab_env.ExperienceBuffer(obs_names, action_names)

    env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/HVAC_Sha_lig_fmu_AB.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV2015.fmu', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_days = 65,
                                 step_size = 900)

    time = pd.date_range(start='1/1/2015', periods=env.n_steps, freq='15min').values
    if env is None:
        quit()
    
    for ep in range(2):
        i = 0
        new_state = env.reset()
        # action_dict = {'QA':[0.0],'yShadeA': [0.0]}
        # action_df =  pd.DataFrame(action_dict) 
        
        for i in range(3360): #35040
            last_action = buffer.last_action()   
            action = test_agent(new_state, last_action)
            new_state, reward, done, _ = env.step(action)
            buffer.append(action,new_state,reward)
            i += 1
            if done:
                actions_data = buffer.action_data()
                obs_data = buffer.obs_data()
                f, axarr = plt.subplots(4)
                axarr[0].plot(time,obs_data['ZoneTempA'])
                axarr[0].set_ylabel('RoomA temp')
                axarr[1].plot(time,obs_data['P_batt'])
                axarr[1].set_ylabel('Battery charging power A')
                axarr[2].plot(time,obs_data['CoolA'])  
                axarr[2].set_ylabel('Cooling load A')
                axarr[3].plot(time,obs_data['LigA'])
                axarr[3].set_ylabel('Lighting A')
                # axarr[3].plot(time,lig)
                # axarr[3].set_ylabel('lig/W')
                # axarr[4].plot(time,mels)
                # axarr[4].set_ylabel('mel/W')
                # axarr[5].plot(time,outdoor_temp)
                # axarr[5].set_ylabel('outdoor temp/degC')
                plt.savefig('results')
                #plt.show()


                break
