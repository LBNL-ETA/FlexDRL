This folder contains the data acquired during the test window, across three experiments. 

## Relevant Dates
All times in America/Los_Angeles time)
* EE experiment:
  * Start: 07/09/2020 11:00:00 
  * End: 07/17/2020 16:59:59
* Shift experiment: 
  * Start: 07/23/2020 10:00:00
  * End: 07/31/2020 23:59:59
* Shed experiment:
  * Start: 08/3/2020 19:00:00
  * End: 08/11/2020 11:00:00

## CSV File headers
Each CSVs are at 1minute intervals. 

1. external.csv
   * time: time in America/Los_Angeles time in ``%Y-%m-%d %H:%M:%S``
   * oat_c: Outdoor air temperature (Celsius)
   * diffused_irradiance_Wm2: Diffused Irradiance (Watt/m2)
   * global_irradiance_Wm2: Global Horizontal Irradiance Irradiance (Watt/m2)
   * pv_generation_scaled_W: PV generation (Watt)

2. cella_state.csv
   * time: time in America/Los_Angeles time in ``%Y-%m-%d %H:%M:%S``
   * ra_plugload_power_W: Power consumption by plug loads (Watts)
   * ra_light_power_W: Power consumption by lights (Watts)
   * ra_fan_power_W: Power consumption by Air Handling Unit supply air fan (Watts)
   * ra_hwp_power_W: Power consumption by Hot Water Pump (Watts)
   * ra_chwp_power_W: Power consumption by Chilled Water Pump (Watts)
   * ra_hw_th_load_W: Power consumption by hot water loop (Watts)
   * ra_chw_th_load_W: Power consumption by chilled water loop (Watts)
   * ra_sup_air_temp_C: Air Handling Unit supply air temperature (Celsius)
   * ra_sup_air_flow_cmh: Air Handling Unit supply air flow rate (cubic meter per hour (cmh))
   * ra_zone_air_temp1_C: Zone Air Temperature from sensor 1 (Celsisus). We used the mean zone air temperature for our calculations.
   * ra_zone_air_temp2_C: Zone Air Temperature from sensor 2 (Celsisus)
   * ra_licor_desk1_1_lux: Indoor illuminance level from desk closest to window on north side (lux)
   * ra_licor_desk2_1_lux: Indoor illuminance level from desk in middle on north side (lux)
   * ra_licor_desk3_1_lux: Indoor illuminance level from desk farthest from window on north side (lux)
   * ra_licor_desk4_1_lux: Indoor illuminance level from desk farthest from window on south side (lux)
   * ra_licor_desk5_1_lux: Indoor illuminance level from desk in middle on south side (lux)
   * ra_licor_desk6_1_lux: Indoor illuminance level from desk closest to window on south side (lux)
   * ra_battery_soc_percentage: CellA battery state of charge (%)

3. cellb_state.csv
   * time: time in America/Los_Angeles time in ``%Y-%m-%d %H:%M:%S``
   * rb_plugload_power_W: Power consumption by plug loads (Watts)
   * rb_light_power_W: Power consumption by lights (Watts)
   * rb_fan_power_W: Power consumption by Air Handling Unit supply air fan (Watts)
   * rb_hwp_power_W: Power consumption by Hot Water Pump (Watts)
   * rb_chwp_power_W: Power consumption by Chilled Water Pump (Watts)
   * rb_hw_th_load_W: Power consumption by hot water loop (Watts)
   * rb_chw_th_load_W: Power consumption by chilled water loop (Watts)
   * rb_sup_air_temp_C: Air Handling Unit supply air temperature (Celsius)
   * rb_zone_air_temp1_C: Zone Air Temperature from sensor 1 (Celsisus). We used the mean zone air temperature for our calculations.
   * rb_zone_air_temp2_C: Zone Air Temperature from sensor 2 (Celsisus)
   * rb_licor_desk1_1_lux: Indoor illuminance level from desk closest to window on north side (lux)
   * rb_licor_desk2_1_lux: Indoor illuminance level from desk in middle on north side (lux)
   * rb_licor_desk3_1_lux: Indoor illuminance level from desk farthest from window on north side (lux)
   * rb_licor_desk4_1_lux: Indoor illuminance level from desk farthest from window on south side (lux)
   * rb_licor_desk5_1_lux: Indoor illuminance level from desk in middle on south side (lux)
   * rb_licor_desk6_1_lux: Indoor illuminance level from desk closest to window on south side (lux)
   * rb_battery_soc_percentage: CellA battery state of charge (%)

4. setpoints.csv
   * time: time in America/Los_Angeles time in ``%Y-%m-%d %H:%M:%S``
   * ra_zone_temp_hsp_C: CellA Zone Temperature Heating Setpoint (Celsius)
   * ra_zone_temp_csp_C: CellA Zone Temperature Cooling Setpoint (Celsius)
   * ra_battery_rate_W: CellA battery charge/discharge rate (Watt)
   * rb_sup_air_flow_sp_cmh: CellB Air Handling Unit Supply Air Flow setpoint (meter3/hour)
   * rb_sup_air_temp_sp_C: CellB Air Handling Unit Supply Air Temperature setpoint (Celsius)
   * rb_battery_rate_W: CellB battery charge/discharge rate (Watt)
