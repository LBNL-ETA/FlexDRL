def get_density(medium, temp_K):
    if medium == "water":
        A = 206.7
        B = 7.01013
        C = -0.0195311
        D = 0.0000164685
    elif medium == "glycol":
        #From Internet (2016/04/08) Propylene Glycol
        A = 1187.6
        B = -0.3789
        C = -0.0003
        D = -0.0000007

    else:
        return 0

    water_density_kgPerCubicMeter = A + B * temp_K + C * pow(temp_K, 2) + D * pow(temp_K, 3)
    return water_density_kgPerCubicMeter

def get_specific_heat_capacity(medium, temp_K):
    if medium == "water":
        A = 9014.9707450016
        B = -41.0478625328587
        C = 0.113903509102131
        D = -0.000102766992663795
    elif medium == "air":
        A = 1036.83233764257
        B = -0.239217492121362
        C = 0.000458498890346236
        D = 0
    elif medium == "vapor":
        A = 1936.0
        B = -0.72
        C = 0.0016
        D = 0
    elif medium == "glycol":
        #From Internet (2016/04/08) Propylene Glycol
        A = 717.05
        B = 6.029
        C = 0.000000000002
        D = -0.000000000000002

    else:
        return 0

    heat_capacity_JPerkgKevin = A + B * temp_K + C * pow(temp_K, 2) + D * pow(temp_K, 3)
    return heat_capacity_JPerkgKevin


def calculater_water_energy_flow_rate(water_flow_LPM, water_sup_temp_C, water_ret_temp_C, hw_or_chw):
    if hw_or_chw == 'chw':
        fraction_water_glycol = 0.9
    else:
        fraction_water_glycol = 1

    water_sup_temp_K = water_sup_temp_C + 273.15
    water_ret_temp_K = water_ret_temp_C + 273.15

    water_avg_temp_K = (water_sup_temp_K + water_ret_temp_K) * 0.5
    water_diff_temp_K = water_sup_temp_K - water_ret_temp_K

    density_kgPerCubicMeter = fraction_water_glycol * get_density("water", water_avg_temp_K) + (
                1 - fraction_water_glycol) * get_density("glycol", water_avg_temp_K)

    mass_flow_rate_kgPerSec = water_flow_LPM * density_kgPerCubicMeter / 1000.0 / 60.0

    heat_capacity_JPerkgKevin = fraction_water_glycol * get_specific_heat_capacity("water", water_avg_temp_K) + (
                1 - fraction_water_glycol) * get_specific_heat_capacity("glycol", water_avg_temp_K)

    power_W = heat_capacity_JPerkgKevin * mass_flow_rate_kgPerSec * water_diff_temp_K

    return power_W