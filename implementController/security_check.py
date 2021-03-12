def security_battery(P_ctrl, SOC, capacity=7, time_step=0.25,
                     upper_bound=1, lower_bound=0.1):
    """
    Security check for the battery control
    :param P_ctrl: kW, control signal for charging power, output from RL controller,
        positive for charging, negative for discharging
    :param SOC: State of Charge
    :param capacity: kWh, capacity of battery, 7kWh as default
    :param timestep: h, timestep for each control, 15min as default
    :param upper_bound: upper bound of the SOC of the battery, 1 as default
    :param lower_bound: lower bound of the SOC of the battery, 0.1 as default
    :return P: output of charging rate of the battery for this time step
    """

    if P_ctrl >= 0: # charging the battery
        P_max = (upper_bound-SOC)*capacity/time_step
        P = min(P_ctrl, P_max)
    else:           # discharging the battery
        P_min = (lower_bound-SOC)*capacity/time_step
        P = max(P_ctrl,P_min)

    return P
