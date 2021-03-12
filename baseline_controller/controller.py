import yaml
from flexlab.db_layer.db_interface import  DB_Interface
import datetime
import pytz
import pandas as pd
import time

# TODO: add shed and shift events and setpoint handling
class Baseline_Controller:
    def __init__(self, config_file="baseline_config.yaml"):
        with open(config_file) as fp:
            self.config = yaml.safe_load(fp)

        self.db_interface = DB_Interface()
        self.max_battery_rate = self.config.get('max_battery_rate', 3300)
        self.min_battery_rate = self.config.get('min_battery_rate', -3300)
        self.max_battery_soc = self.config.get('max_battery_soc', 90)
        self.min_battery_soc = self.config.get('min_battery_soc', 11)
        self.min_air_flow = self.config.get('min_air_flow_kgs', 0.05)

        self.tz_local = pytz.timezone("America/Los_Angeles")
        self.tz_utc = pytz.timezone("UTC")
        self.current_cell = self.config.get('current_cell', 'a')
        self.battery_total_capacity = self.config.get('battery_total_capacity', 7200)

    def _get_light_sp(self, time_now):
        hour_now = time_now.hour
        if hour_now>=7 and hour_now<20:
            light_sp = 1
        else:
            light_sp = 0
        return light_sp

    def _get_battery_sp(self, df):
        """
        "Generate sample battery setpoints using 2017 data and ensure that the battery behaves as expected:
        - EE strategy: If PV > building_load, charge the rest to the battery.
                        If PV < building_load, discharge the battery to support the building load as well (whenever PV is insufficient)
                        If the battery isn't fully charged at 12AM, charge the battery from 12AM-6AM (when the prices are the cheapest) to be ready for the next day.
        - Shed strategy: During high prices discharge the battery to 20% min; After the high price window recharge the battery to 50% or 90% if  solar power is available (if solar > building load)
        - Shift strategy: During low prices charge the battery up to 90%; During medium prices, charge if solar power is available or if the battery charge is below 50%; During high prices: discharge the battery to 20% min"
        :param df: list of observartions from db_interface.get_data() function
        :return:
        """
        pv_generation_W = df['PV'].values[0]
        battery_soc = df['SOC'].values[0]*100
        building_power_W = df.net_power.values[0]*1000
        net_load = building_power_W - pv_generation_W

        if net_load < 0:
            # PV more than load, charge battery
            print("more PV than load, charge the battery")
            battery_sp = min(-1*net_load, self.max_battery_rate)
        else:
            # PV less than load, discharge the battery to support the load
            battery_sp = max(-1*net_load, self.min_battery_rate)

        if battery_sp > 0:
            if battery_soc >= self.max_battery_soc:
                print("battery fully charged, changing setpoint to 0 from {0}W".format(battery_sp))
                battery_sp = 0
        elif battery_sp < 0:
            if battery_soc <= self.min_battery_soc:
                print("battery empty, changing setpoint to 0W from {0}W".format(battery_sp))
                battery_sp = 0
        else:
            battery_sp = 0

        return battery_sp

    def _get_flexlab_sup_air_flow_sp(self, time_now):
        hour_now = time_now.hour
        if hour_now >=7 and hour_now<19:
            sup_air_flow_sp = -1
        else:
            sup_air_flow_sp = self.min_air_flow
        return sup_air_flow_sp

    def get_actions(self, et=None):
        if et is None:
            et = self.tz_utc.localize(datetime.datetime.utcnow()).astimezone(self.tz_local).replace(tzinfo=None)
        st = et - datetime.timedelta(minutes=14)
        obs = self.db_interface.get_data_controller(st=st, et=et, cell='a')
        print("observations at time = {0} are:".format(et.strftime(("%Y-%m-%d %H:%M:%s"))))
        print(obs[['SOC', 'PV', 'net_power']])
        battery_sp = self._get_battery_sp(df = obs)
        light_sp = self._get_light_sp(time_now=et)
        sup_air_flow_sp = self._get_flexlab_sup_air_flow_sp(time_now=et)

        setpoints_df = pd.DataFrame(index=[et], data={'sup_air_flow_sp': [sup_air_flow_sp], 'sup_air_temp_sp': [-1],
                                                      'light_level_sp': [light_sp], 'battery_sp': [battery_sp]})
        return setpoints_df

    def push_actions(self, setpoints):
        return self.db_interface.push_setpoints_to_db(cell=self.current_cell, df=setpoints)

if __name__ == "__main__":
    controller = Baseline_Controller(config_file="baseline_controller/baseline_config.yaml")
    completed_minute = -1
    printed_minute = -1
    while True:
        time_now = datetime.datetime.now()
        minute_now = time_now.minute
        if minute_now % 15 == 0 and minute_now != completed_minute:
            # actions = controller.get_actions(et=datetime.datetime(2020, 5, 15, 14, 0, 0, 0))
            try:
                actions = controller.get_actions()
                print("actions at {0}".format(time_now))
                print(actions)

                print("pushing actions to FL interface")
                print("push status= {0}".format(controller.push_actions(setpoints=actions)))
                print("\n")
                completed_minute = minute_now
                printed_minute = minute_now
                time.sleep(1)
            except Exception as e:
                print("Error occured while running controller. error = {0}".format(e))
        elif minute_now % 1 == 0 and printed_minute != minute_now:
            print("current time: {0}; waiting for the next 15th minute".format(time_now))
            printed_minute = minute_now
