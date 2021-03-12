import yaml
import requests
import datetime
import pandas as pd
import subprocess
from sqlalchemy import create_engine
import time
import argparse

class FL_Control_Interface:
    def __init__(self, config_file="flexlab_control_config.yaml"):
        with open(config_file, 'r') as fp:
            self.config = yaml.safe_load(fp)

        self.flexgrid_config = self.config.get('flexgrid', {})
        self.flexgrid_control_flag = self.flexgrid_config .get('control_flag', {'ra': False, 'rb': False})
        self.flexgrid_api_config = self.flexgrid_config.get('api', {})

        self.flexlab_config = self.config.get('flexlab', {})
        self.flexlab_control_flag = self.flexlab_config .get('control_flag', {'ra': False, 'rb': False})
        self.cws_point_map = self.flexlab_config.get('cws_point_map', {'rb_sup_air_flow_sp_CMH': 'RB-SAFsp', 'rb_sup_air_temp_sp_C': 'RB-SATsp'})
        self.flexlab_api_config = self.flexlab_config.get('api', {})
        self.flex_user = self.flexlab_api_config.get('flex_user', 'btmder')
        self.flex_password = self.flexlab_api_config.get('flex_password')
        self.flexq_login = self.flexlab_api_config.get('flexq_login', 'queryxr')

        self.lights_config = self.config.get('lights', {})
        self.lights_control_flag = self.lights_config.get('control_flag', {'ra': False, 'rb': False})
        self.lights_api_config = self.lights_config.get('api', {})
        self.lights_host = self.lights_api_config.get('host')
        self.lights_app_id = self.lights_api_config.get('app_id')
        self.lights_app_key = self.lights_api_config.get('app_key')
        self.cree_space_uuid_map = self.lights_config.get('cree_space_uuid_map')

        self.current_battery_setpoint = self.flexgrid_config.get('initial_battery_setpoint', 0)
        self.current_saf_setpoint = self.flexlab_config.get('initial_saf_sp', -1)
        self.current_sat_setpoint = self.flexlab_config.get('initial_sat_sp', -1)
        self.current_lights_setpoint = self.flexgrid_config.get('initial_lights_setpoint', 0)

        self.control_config = self.config.get('control', {})
        self.variables = self.control_config.get('variable_names', [])
        self.query_time = self.control_config.get('query_time_minutes', 16)
        self.time_threshold = self.control_config.get('revert_to_default_time_threshold_seconds', 3600)
        self.default_setpoints = self.control_config.get('default_values')

        self.database_config = self.config.get('database')
        self.initialize_db_client()

    def initialize_db_client(self):
        self.host = self.database_config.get('host', 'localhost')
        self.port = self.database_config.get('port', 9003)
        self.database = self.database_config.get('database', 'btm_drl')
        self.username = self.database_config.get('username', 'btmdrl')
        self.password = self.database_config.get('password')
        self.setpoints_table = self.database_config.get('setpoints_table', 'setpoints')


        url = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(self.username, self.password, self.host,
                                                        self.port, self.database)
        self.engine = create_engine(url)
        self.client = self.engine.connect()


    def get_latest_setpoints(self):
        """
        function to query latest setpoints from database

        :return: a list of dictionaries, with each dictionary containing two keys: time and the corresponding
        setpoint variable name
        """
        df_list = []
        for variable in self.variables:
            query = "select * from {0} where time in (select max(time) from {1} where time > now() - interval '{2} minutes' and name = '{3}') and name = '{4}';".format(
                self.setpoints_table, self.setpoints_table, self.query_time, variable, variable)
            df = pd.read_sql(query, self.client, index_col=['time'], coerce_float=True, parse_dates=True,
                             columns=['name', 'value'])
            df = df.pivot(columns='name', values='value')
            df_list.append(df)

        latest_df = pd.concat(df_list, axis=1)
        latest_setpoints = []
        for col in latest_df.columns:
            col_df = latest_df[[col]].dropna()
            latest_setpoint = {'time': col_df.index.values[0], 'variable': col, 'value': col_df[col].values[0]}
            latest_setpoints.append(latest_setpoint)
        return latest_setpoints

    # Periodic function that sets setpoints to devices
    def set_setpoints(self):
        latest_setpoints = self.get_latest_setpoints()
        time_now = datetime.datetime.utcnow()
        for setpoint in latest_setpoints:
            setpoint_time = datetime.datetime.utcfromtimestamp(setpoint.get('time').astype('O')/1e9)
            variable = setpoint.get('variable')
            new_value = setpoint.get('value')

            if (time_now - setpoint_time).total_seconds() >= self.time_threshold:
                # if the setpoint was generated more than a hour ago, revert to default setpoint
                new_value = self.default_setpoints[variable]
            elif time_now < setpoint_time:
                # if the setpoint has been generated for a future time, revert to default setpoint
                new_value = self.default_setpoints

            cell = variable[1]

            if variable.endswith("_sup_air_flow_sp_cmh_command") or variable.endswith(
                    "_sup_air_temp_sp_command") or variable.endswith(
                    "_zone_temp_hsp_command") or variable.endswith(
                    "_zone_temp_csp_command"):
                if self.flexlab_control_flag.get('r'+cell):
                    ret = self.set_flexlab_points(cell=variable[1], point_name=variable, value=new_value)
                else:
                    print("flexlab_control_flag for cell r{0} is set to False. Not changing setpoints".format(cell))
                    ret = False
            elif variable.endswith("_battery_sp_command"):
                if self.flexgrid_control_flag.get('r' + cell):
                    ret = self.set_battery_rate(cell=variable[1], value=new_value)
                else:
                    print("flexgrid_control_flag for cell r{0} is set to False. Not changing setpoints".format(cell))
                    ret = False
            elif variable.endswith("_light_level_sp_command"):
                if self.lights_control_flag.get('r' + cell):
                    if new_value < 0.01:
                        print("rounding r{0}_light_sp to 0".format(cell))
                        new_value = 0
                    ret = self.set_light_level(cell=variable[1], value=new_value)
                else:
                    print("lights_control_flag for cell r{0} is set to False. Not changing setpoints".format(cell))
                    ret = False
            else:
                ret = False

            if ret:
                print("At time {0} UTC succesfully set value={1} to variable={2}".format(time_now, new_value, variable))
            else:
                print("failed to set value={0} to variable={1}".format(new_value, variable))
        if len(latest_setpoints) == 0:
            print("no setpoints retrieved")

    def set_battery_rate(self, cell, value):
        if cell != 'a' and cell != 'b':
            print("Wrong cell={0}".format(cell))
            return False

        if value > 3300 or value < -3300:
            print("Rate={0} out of bounds. Rate must be between -3300W and 3300W".format(value))
            return False

        if cell == 'a':
            inverter_id = 1
        else:
            inverter_id = 3

        url = self.flexgrid_api_config.get('host', 'http://flexgrid-s1.dhcp.lbl.gov') + ":{0}/control?".format(self.flexgrid_api_config.get('port', '9090'))
        url += "inv_id={0},Batt_ctrl={1}".format(inverter_id, value)

        try:
            resp = requests.get(url, verify=False)
            if resp.status_code == 200:
                self.current_battery_setpoint = value
                print("Flexgrid API response = {0}".format(resp.content))
                return True
            else:
                print("error sending battery charge/discharge rate of {0} to inverter {1}".format(value, inverter_id))
                return False
        except:
            print("request to Flexgrid API failed, not setting setpoint for cell = {0}".format(cell))
            return False

    def set_flexlab_points(self, cell, point_name, value):
        flex_sys = "Cell HVAC and Perm Sensors R{0}".format(cell.upper())

        channel = self.cws_point_map.get(point_name)

        set_cmd = "ssh %s@flexq.lbl.gov \'{\"cmd\":\"SETDAQ\", \"sys\":\"%s\", \"chn\":\"%s\", \"val\":\"%f\", \"user\":\"%s\",\"pass\":\"%s\"}\'"
        print(set_cmd % (self.flexq_login, flex_sys, channel, value, self.flex_user, self.flex_password))

        try:
            subprocess.check_output(set_cmd % (self.flexq_login, flex_sys, channel, value, self.flex_user, self.flex_password), shell=True)
            return True
        except Exception as e:
            print("exception occurred when setting {0} to point {1} in cell {2}, error={3}".format(value, point_name, cell, str(e)))
            return False

    def set_light_level(self, cell, value):
        if cell != 'a' and cell != 'b':
            print("Wrong cell={0}".format(cell))
            return False

        if value > 1 or value < 0:
            print("light level ={0} out of bounds. Level must be between 0 and 100".format(value))
            return False

        headers = {
            'accept': 'application/json',
            'APPID': self.lights_app_id,
            'APIKEY': self.lights_app_key,
            'content-type': 'application/json'
        }

        space_uuid = self.cree_space_uuid_map.get('r{0}'.format(cell), None)
        if not space_uuid is None:
            url = "{0}/smartcast-api/v1/spaces/{1}/actuator".format(self.lights_host, space_uuid)
            value_to_be_written = int(value*100)
            data = {'light_level': value_to_be_written}
            try:
                resp = requests.post(url=url, headers=headers, json=data, verify=False)
                if resp.status_code == 201:
                    ret = True
                else:
                    print("error while trying to change light_level of space=r{0}, uuid={1} to value={2}, response code = {3}".format(cell, space_uuid, value, resp.status_code))
                    ret = False
            except:
                print("request to Cree API failed, not setting setpoint for cell = r{0}".format(cell))
                ret = False
            return ret


if __name__ == "__main__":
    controller = FL_Control_Interface(config_file="flexlab_control_config.yaml")
    parser = argparse.ArgumentParser()

    parser.add_argument("--reset_cella", help="reset cell a setpoints", action="store_true", default=False)
    parser.add_argument("--reset_cellb", help="reset cell b setpoints", action="store_true", default=False)
    parser.add_argument("--reset_both", help="reset cells a and b setpoints", action="store_true", default=False)
    args = parser.parse_args()

    if args.reset_both:
        print("resetting both cells")
        controller.set_battery_rate(cell='a', value=0)
        controller.set_battery_rate(cell='b', value=0)

        controller.set_flexlab_points(cell='a', point_name='ra_sup_air_flow_sp_cmh_command', value=-1)
        # controller.set_flexlab_points(cell='a', point_name='ra_sup_air_temp_sp', value=-1)
        controller.set_flexlab_points(cell='a', point_name='ra_zone_temp_hsp_command', value=-1)
        controller.set_flexlab_points(cell='a', point_name='ra_zone_temp_csp_command', value=-1)
        controller.set_flexlab_points(cell='b', point_name='rb_sup_air_flow_sp_cmh_command', value=-1)
        controller.set_flexlab_points(cell='b', point_name='rb_sup_air_temp_sp_command', value=-1)

        time_now = datetime.datetime.now()
        if time_now.hour >= 7 and time_now.hour <= 7:
            light_sp = 100
        else:
            light_sp = 0
        controller.set_light_level(cell='a', value=light_sp)
        controller.set_light_level(cell='b', value=light_sp)
    elif args.reset_cella:
        print("resetting cell a")
        controller.set_battery_rate(cell='a', value=0)

        controller.set_flexlab_points(cell='a', point_name='ra_sup_air_flow_sp_cmh_command', value=-1)
        # controller.set_flexlab_points(cell='a', point_name='ra_sup_air_temp_sp', value=-1)
        controller.set_flexlab_points(cell='a', point_name='ra_zone_temp_hsp_command', value=-1)
        controller.set_flexlab_points(cell='a', point_name='ra_zone_temp_csp_command', value=-1)

        time_now = datetime.datetime.now()
        if time_now.hour >= 7 and time_now.hour <= 7:
            light_sp = 100
        else:
            light_sp = 0

        controller.set_light_level(cell='a', value=light_sp)
    elif args.reset_cellb:
        print("resetting cell b")
        controller.set_battery_rate(cell='b', value=0)

        controller.set_flexlab_points(cell='b', point_name='rb_sup_air_flow_sp_cmh_command', value=-1)
        controller.set_flexlab_points(cell='b', point_name='rb_sup_air_temp_sp_command', value=-1)

        time_now = datetime.datetime.now()
        if time_now.hour >= 7 and time_now.hour <= 7:
            light_sp = 100
        else:
            light_sp = 0

        controller.set_light_level(cell='b', value=light_sp)
    else:
        prev_minute = -1
        while True:
            time_now = datetime.datetime.utcnow()
            if prev_minute != time_now.minute:
                try:
                    print("getting latest setpoints")
                    controller.set_setpoints()
                    prev_minute = time_now.minute
                    print("\n")
                except Exception as e:
                    print("Error occurred while setting setpoints. Error={0}".format(e))
            elif time_now.second%30 == 0:
                print("current time = "+datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")+" UTC")
                time.sleep(1)
