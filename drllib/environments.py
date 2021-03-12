from gym_flexlab.envs import flexlab_env


def set_envs(sim_days, eprice_ahead, alpha_r, beta_r, gamma_r, pv_panels, light_ctrl, delta_r=0.0):

    """
    Function used to set up the environement used in the training process
    """
    env1 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2013.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2013.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2013_shed.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2013.csv',
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2013,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = sim_days,
                                 step_size = 900,
                                 eprice_ahead = eprice_ahead,
                                 alpha_r = alpha_r,
                                 beta_r = beta_r,
                                 gamma_r = gamma_r,
                                 delta_r = delta_r,
                                 pv_panels = pv_panels,
                                 light_ctrl = light_ctrl)

    env2 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2014.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2014.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2014_shed.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2014.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2014,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = sim_days,
                                 step_size = 900,
                                 eprice_ahead = eprice_ahead,
                                 alpha_r = alpha_r,
                                 beta_r = beta_r,
                                 gamma_r = gamma_r,
                                 delta_r = delta_r,
                                 pv_panels = pv_panels,
                                 light_ctrl = light_ctrl)

    env3 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2015.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2015.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2015_shed.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2015.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2015,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = sim_days,
                                 step_size = 900,
                                 eprice_ahead = eprice_ahead,
                                 alpha_r = alpha_r,
                                 beta_r = beta_r,
                                 gamma_r = gamma_r,
                                 delta_r = delta_r,
                                 pv_panels = pv_panels,
                                 light_ctrl = light_ctrl)

    env4 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_TMY.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_TMY.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2017_shed.csv',
                                 daylight_path= 'daylighting/daylight_SFO_TMY.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = sim_days,
                                 step_size = 900,
                                 eprice_ahead = eprice_ahead,
                                 alpha_r = alpha_r,
                                 beta_r = beta_r,
                                 gamma_r = gamma_r,
                                 delta_r = delta_r,
                                 pv_panels = pv_panels,
                                 light_ctrl = light_ctrl)

    # env5 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_Miami_TMY.fmu',
    #                              battery_path = 'fmu_models/battery.fmu',
    #                              pv_path =  'fmu_models/PV_FMU/PV_Miami_TMY.fmu',
    #                              eprice_path = 'e_tariffs/e_d_price_2017.csv',
    #                              daylight_path= 'daylighting/daylight_Miami_TMY.csv', 
    #                              chiller_COP = 3.0, 
    #                              boiler_COP = 0.95,
    #                              sim_year = 2017,
    #                              tz_name = 'America/Los_Angeles',
    #                              sim_days = sim_days,
    #                              step_size = 900,
    #                              eprice_ahead = eprice_ahead,
    #                              alpha_r = alpha_r,
    #                              beta_r = beta_r,
    #                              gamma_r = gamma_r,
    #                              delta_r = delta_r,
    #                              pv_panels = pv_panels,
    #                              light_ctrl = light_ctrl)

    # env6 = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_Chicago_TMY.fmu',
    #                              battery_path = 'fmu_models/battery.fmu',
    #                              pv_path =  'fmu_models/PV_FMU/PV_Chicago_TMY.fmu',
    #                              eprice_path = 'e_tariffs/e_d_price_2017.csv',
    #                              daylight_path= 'daylighting/daylight_Chicago_TMY.csv', 
    #                              chiller_COP = 3.0, 
    #                              boiler_COP = 0.95,
    #                              sim_year = 2017,
    #                              tz_name = 'America/Los_Angeles',
    #                              sim_days = sim_days,
    #                              step_size = 900,
    #                              eprice_ahead = eprice_ahead,
    #                              alpha_r = alpha_r,
    #                              beta_r = beta_r,
    #                              gamma_r = gamma_r,
    #                              delta_r = delta_r,
    #                              pv_panels = pv_panels,
    #                              light_ctrl = light_ctrl)    


    test_env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_SFO_2017.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2017.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2017_shed.csv',
                                 daylight_path= 'daylighting/daylight_SFO_2017.csv',  
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = sim_days,
                                 step_size = 900,
                                 eprice_ahead = eprice_ahead,
                                 alpha_r = alpha_r,
                                 beta_r = beta_r,
                                 gamma_r = gamma_r,
                                 delta_r = delta_r,
                                 pv_panels = pv_panels,
                                 light_ctrl = light_ctrl)

    env =[env1,env2,env3,env4]#,env5,env6]

    return env, test_env



def set_pred_env(sim_days, eprice_ahead, alpha_r, beta_r, gamma_r, pv_panels, light_ctrl, delta_r=0.0):

    """
    Function used to set up the environement used in the prediction process
    """

    env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v3/FlexlabXR_v3_flexlab_2020_new.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_flexlab_2020.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2020_shed_new.csv',
                                 daylight_path= 'daylighting/daylight_SFO_TMY.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2014,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = sim_days,
                                 step_size = 900,
                                 eprice_ahead = eprice_ahead,
                                 alpha_r = alpha_r,
                                 beta_r = beta_r,
                                 gamma_r = gamma_r,
                                 delta_r = delta_r,
                                 pv_panels = pv_panels,
                                 light_ctrl = light_ctrl)

    return env