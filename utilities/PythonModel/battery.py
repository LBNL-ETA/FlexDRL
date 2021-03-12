# -*- coding: utf-8 -*-
"""
Battery model

"""

class Battery:
    '''
    Parameter
    ------------------------------------------
    Ecap: battery capacity [J]
    P_cap_charge: charging capacity [W]
    P_cap_discharge: discharging capacity [W]
    eta_charge: charging efficiency [1]
                defined as power_charged/power_input
    eta_discharge: discharging efficiency [1]
                defined as power_output/power_discharged
    
    Input
    ------------------------------------------
    a: control signal, [0,1] for charging 
                       [-1,0] for discharging
    duration: duration of timestep [s]
    
    Output
    ------------------------------------------    
    SOC: state of charge of the battery [1], 0 for not charged
                                             1 for fully charged
    Preal: real power input of the battery [W], positive for charging
                                                negative for discharging    
    '''
    
    def __init__(self, Ecap=7*3600000, P_cap_charge=3300, P_cap_discharge=3300, eta_charge=0.96, eta_discharge=0.96):
        self.Ecap = Ecap
        self.P_cap_charge = P_cap_charge
        self.P_cap_discharge = P_cap_discharge
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.E = 0   # initial battery energy level is 0
    
    def charge(self,a,duration):
        self.Preal = a*self.P_cap_charge     
        self.E_0 = self.E                               # initial state
        self.E = self.E_0 + self.eta_charge*self.Preal*duration
             # battery will absorb less energy than real power input
        if self.E > self.Ecap:                          # fully charged
            self.E = self.Ecap
        self.SOE = self.E/self.Ecap
        self.Preal = (self.E-self.E_0)/(duration*self.eta_charge)
        return self.SOE, self.Preal
        
    def discharge(self,a,duration):
        self.Preal = a*self.P_cap_discharge
        self.E_0 = self.E                               # initial state
        self.E = self.E_0 + self.Preal*duration/self.eta_discharge  
             # battery will lose more energy than real power output
        if self.E < 0:
            self.E = 0                                  # fully discharged
        self.SOE = self.E/self.Ecap
        self.Preal = (self.E-self.E_0)*self.eta_discharge/duration
        return self.SOE, self.Preal
        
    def battery_interact(self,a,duration):
        if a >= 0:
            return self.charge(a,duration)
        else:
            return self.discharge(a,duration)
