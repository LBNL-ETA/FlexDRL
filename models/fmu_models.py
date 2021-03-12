# -*- coding: utf-8 -*-
"""

@author: Samir Touzani
"""
from pyfmi import load_fmu
import numpy as np

import pandas as pd


class FmuModel(object):

    def __init__(self,
                 fmu_model_path):

        # load the fmu model
        self.model = load_fmu(fmu_model_path,kind='cs', log_level = 0)

    
    def set_parameters(self,
                       parameters):
        # set up model parameters
        for key, value in parameters.items():
            self.model.set(key,value)
        

    def create_fmu(self,
                   start_time,
                   final_time):
        
        # initialize the model
        self.model.initialize(start_time, final_time)

    def simulate(self, 
                 action_names,
                 action,
                 output_names,
                 time_steps,
                 step_size,
                 time_step_idx):

        
        # set action 
        # model.set(['u1','u2','u3'],[inp,inp,inp])
        self.model.set(action_names, action)

        # simulate smu
        self.model.do_step(current_t = time_steps[time_step_idx],
                           step_size = step_size,
                           new_step = True)

        # extract the observations

        obs = np.concatenate(self.model.get(output_names)) #np.array([self.model.get(output_names)])
        

        return obs

    def unload_fmu(self):
        #del self.model
        self.model.reset()

    def terminate_fmu(self):
        self.model.terminate()


