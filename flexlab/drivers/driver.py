from sqlalchemy import create_engine
import yaml
import datetime
import pandas as pd
import numpy as np
import requests
import json
import subprocess

class Driver:
    def __init__(self, config_file):
        with open(config_file, "r") as fp:
            self.config = yaml.safe_load(fp)
        self.database_config = self.config.get('database')
        self.initialize_db_client()

    def initialize_db_client(self):
        self.host = self.database_config.get('host', 'localhost')
        self.port = self.database_config.get('port', 9003)
        self.database = self.database_config.get('database', 'btm_drl')
        self.username = self.database_config.get('username', 'btmdrl')
        self.password = self.database_config.get('password')

        url = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(self.username, self.password, self.host,
                                                        self.port, self.database)
        self.engine = create_engine(url)
        self.client = self.engine.connect()

    def push_to_db(self, df, table_name):
        df_list = []
        for col in df.columns:
            df2 = df[[col]]
            df2.columns = ['value']
            df2['name'] = col
            df_list.append(df2)
        final_df = pd.concat(df_list, axis=0).reset_index()

        query = 'insert into {0} values '.format(table_name) + ','.join(final_df.apply(
            lambda x: "('{0}', '{1}', {2})".format(x['time'].strftime("%Y-%m-%d %H:%M:%S"), x['name'], x['value']),
            axis=1).values) + ' on conflict (time, name) do update ' + 'SET value = excluded.value;'
        try:
            self.client.execute(query)
            print("pushed to table {0}".format(table_name))
        except Exception as e:
            print("error occurred while pushing to table {0}, error={1}".format(table_name, str(e)))


    def get_utc_time_now(self):
        return datetime.datetime.utcnow()


    def get_data_smap(self, url, point_uuid_map, time_now, time_threshold=60):
        start = time_now - datetime.timedelta(minutes=time_threshold)
        end = time_now

        df_list = []
        for uuid in point_uuid_map:
            point_name = point_uuid_map[uuid]
            query = "select data in ({0}000, {1}000) where uuid = \"{2}\"".format(start.strftime("%s"),
                                                                                  end.strftime("%s"), uuid)
            rsp = requests.post(url, data=query)
            data = json.loads(rsp.content.decode("utf-8"))

            data = np.array(data[0]['Readings'])

            if len(data) > 0:
                values = data[:, 1]
                df = pd.DataFrame(data={point_name: values, 'time': data[:, 0] / 1000})
                df.time = pd.to_datetime(df.time, unit='s')
                df.set_index('time', inplace=True)
                df_list.append(df)
        if len(df_list) > 0:
            df = pd.concat(df_list, axis=1)
            return df
        return pd.DataFrame()


    def get_data_cws(self, flexq_login, fl_username, fl_password):
        cmd = "ssh %s@flexq.lbl.gov \'{\"cmd\":\"GETUSERDATA\",\"user\":\"\'%s\'\",\"pass\":\"\'%s\'\",\"cmdslp\":1.0,\"rcvsz\":165536}\' "%(flexq_login, fl_username, fl_password)
        try:
            op = subprocess.check_output(cmd, shell=True)
            point_list = json.loads(op)
        except Exception as e:
            print("error occured while trying query flexq, error={0}".format(str(e)))
            point_list = []
        return point_list

    def filter_cws_data(self, cws_point_list, point_map, time_now):
        data = {}

        for point in cws_point_list:
            if point != '':
                split_content = point.split(':')
                flexq_point_name = split_content[1]
                point_name = point_map.get(flexq_point_name, None)
                if point_name != None:
                    value = float(split_content[3])
                    if point_name.endswith("lux"):
                        value = max(0, value)
                    data[point_name] = [value]
        if len(data) == 0:
            return pd.DataFrame()

        air_flow_points_list = ['ra_sup_air_flow_cmh', 'ra_sup_air_flow_sp_cmh', 'rb_sup_air_flow_cmh', 'rb_sup_air_flow_sp_cmh']
        for air_flow_point in air_flow_points_list:
            if air_flow_point in data:
                sup_air_temp_point = air_flow_point.replace('flow', 'temp').replace('cmh', 'C')
                new_point_name = air_flow_point.split('_cmh')[0] + '_kgs'
                if data[air_flow_point][0] == -1:
                    if air_flow_point in ['ra_sup_air_flow_sp_cmh', 'rb_sup_air_flow_sp_cmh']:
                        data[new_point_name] = [-1]
                else:
                    if sup_air_temp_point in data:
                        new_value = self._convert_sup_air_flow_to_kgs_from_cmh(sup_air_flow_cmh=data[air_flow_point][0],
                                                                               sup_air_temp_C=data[sup_air_temp_point][0])
                        data[new_point_name] = [new_value]


        data['time'] = [time_now]
        df = pd.DataFrame(data)
        df = df.set_index('time')

        return df

    def _get_density_kgm3(self, sup_air_temp_C):
        z = np.poly1d([-3.25360845e-08, 4.37156086e-05, -2.14726928e-02, 4.56125958e+00])
        sup_air_temp_K = sup_air_temp_C + 273.15
        return z(sup_air_temp_K)

    def _convert_sup_air_flow_to_kgs_from_cmh(self, sup_air_flow_cmh, sup_air_temp_C):
        density_kgm3 = self._get_density_kgm3(sup_air_temp_C=sup_air_temp_C)
        return sup_air_flow_cmh * density_kgm3 / 3600.0