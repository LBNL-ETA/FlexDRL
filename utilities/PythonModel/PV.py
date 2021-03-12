# -*- coding: utf-8 -*-
"""
PV model

"""

class PV:
    '''
    Parameter
    ------------------------------------------
    A: PV array area [m2]
    eff: PV panel efficiency [1]
    effDcAc: inverter efficiency [1]

    Input
    ------------------------------------------
    linc: Solar irradiation incident on array [W/m2]

    Output
    ------------------------------------------
    Pgen: Power generated by the array [W]
    '''

    def __init__(self, A, eff=0.2, effDcAc=0.8):
        self.A = A
        self.eff = eff
        self.effDcAc = effDcAc

    def generate(self,linc):
	assert linc>=0, 'Solar irradiation should be positive'       
	self.power_generated = linc*self.A*self.eff*self.effDcAc
        return self.power_generated
