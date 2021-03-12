import yaml
from sqlalchemy import create_engine
import pandas as pd
import pytz
import datetime
import numpy as np


class DB_Interface:
    def __init__(self, config_file="flexlab/db_layer/db_interface_config.yaml"):
        with open(config_file, 'r') as fp:
            self.config = yaml.safe_load(fp)

        self.tz_local = pytz.timezone("America/Los_Angeles")
        self.tz_utc = pytz.timezone("UTC")

        self.database_config = self.config.get('database')
        self.setpoint_table = self.database_config.get('setpoint_table', 'setpoints')

        self.post_process_config = self.config.get('post_process')

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

    def _query_database(self, query):
        try:
            df = pd.read_sql(query, self.client, index_col=['time'], coerce_float=True, parse_dates=True,
                             columns=['name', 'value'])
            df = df.pivot(columns='name', values='value')

            # convert UTC time to local time DatetimeIndex
            df = df.tz_localize(self.tz_utc).tz_convert(self.tz_local).tz_localize(None)
            return df
        except Exception as e:
            print("error occurred while executing query {0}, error={1}".format(query, str(e)))

    def _query_single_system(self, st, et, cell, system, resample_minutes):
        if not cell.startswith('r'):
            cell = 'r{0}'.format(cell)

        # Assume time is in localtime; convert into UTC
        if st.tzinfo is None:
            st_str = self.tz_local.localize(st).astimezone(self.tz_utc).strftime("%Y-%m-%d %H:%M:%S")
        else:
            st_str = st.astimezone(self.tz_utc).strftime("%Y-%m-%d %H:%M:%S")

        # Assume time is in localtime; convert into UTC
        if et.tzinfo is None:
            et_str = self.tz_local.localize(et).astimezone(self.tz_utc).strftime("%Y-%m-%d %H:%M:%S")
        else:
            et_str = et.astimezone(self.tz_utc).strftime("%Y-%m-%d %H:%M:%S")

        system_details = self.config.get('query_condition').get(cell).get(system)
        variable_agg_map = system_details.get('variable_agg_map', None)
        if variable_agg_map is None:
            variables = system_details.get('variables')
            agg = system_details.get('agg', 'last')
            agg_list = [agg] * len(variables)
            variable_agg_map = dict(zip(variables, agg_list))
        else:
            variables = variable_agg_map.keys()

        table = system_details.get('table')
        query = "select * from {0} where time >= '{1}' and time <='{2}'".format(table, st_str, et_str)
        query_condition = ' and (name = ' + ' or name = '.join(["'{0}'".format(var) for var in variables]) + " );"

        query = query + query_condition
        df = self._query_database(query)
        resample = '{0}T'.format(resample_minutes)
        df = df.resample(resample, label='right', closed='right').agg(variable_agg_map)
        return df

    def _format_execute_query_result(self, result):
        output = result.fetchall()
        output = [op[0] for op in output]
        return output

    def get_columns(self, cell='a', table='all'):
        table_list = []
        if table == 'all':
            for system in self.config.get('query_condition').get(cell):
                system_details = self.config.get('query_condition').get(cell).get(system)
                table = system_details.get('table')
                table_list.append(table)
        else:
            table_list = [table]

        all_columns = []
        for table_name in table_list:
            result = self.client.execute("select distinct name from {0};".format(table_name))
            res = self._format_execute_query_result(result=result)
            all_columns = all_columns + res

        filtered_columns = []
        for col in all_columns:
            if col.startswith('r'+cell) or col == "pv_generation_W" or col == "pv_generation_scaled_W":
                filtered_columns.append(col)
        return filtered_columns


    def get_data(self, st, et, cell='b', resample_minutes=15, include_setpoints=False):
        """
        get data for a whole cell for FL
        :param st: datetime.datetime local time
        :param et: datetime.datetime local time
        :param cell: 'a' or 'b'
        :param resample_minutes: data frequency; default 15; minimum 1
        :param include_setpoints: flag to specify if generated setpoints should be queried or not
        :return: dataframe with each column a separate variable with a DatetimeIndex in local time
        """

        df_list = []
        cell = 'r{0}'.format(cell)
        for system in self.config.get('query_condition').get(cell):
            if system == 'setpoints' and not include_setpoints:
                continue

            df = self._query_single_system(st=st, et=et, cell=cell, system=system,
                                           resample_minutes=resample_minutes)
            df_list.append(df)

        final_df = pd.concat(df_list, axis=1)
        return final_df

    def get_setpoints(self, st, et, cell='b', resample_minutes=15):
        """
        get data for a whole cell for FL
        :param st: datetime.datetime local time
        :param et: datetime.datetime local time
        :param cell: 'a' or 'b'
        :param resample_minutes: data frequency; default 15; minimum 1
        :return: dataframe with each column a separate variable with a DatetimeIndex in local time
        """

        cell = 'r{0}'.format(cell)
        system = 'setpoints'
        df = self._query_single_system(st=st, et=et, cell=cell, system=system,
                                       resample_minutes=resample_minutes)
        return df

    def get_data_controller(self, st, et, cell='b', resample_minutes=15):
        df = self.get_data(st=st, et=et, cell=cell, resample_minutes=resample_minutes, include_setpoints=False)
        processed_df = self._post_process(raw_df=df, cell=cell, resample_time=resample_minutes)
        return processed_df

    def _post_process(self, raw_df, cell, resample_time):
        if not cell.startswith('r'):
            cell = 'r{0}'.format(cell)

        df = raw_df.copy(deep=True)
        df['TOD'] = df.index.hour + df.index.minute / 60.0
        cell_post_process_config = self.post_process_config.get(cell)

        post_processed_variables1 = list(cell_post_process_config.keys())
        post_processed_variables2 = post_processed_variables1

        num_new_variables = len(post_processed_variables1)
        while num_new_variables > 0:
            for var in post_processed_variables1:
                var_config = cell_post_process_config.get(var)
                requires = var_config.get('requires', [])
                missing_required_var = False
                for required_var in requires:
                    if required_var not in df.columns:
                        missing_required_var = True

                if missing_required_var:
                    continue
                variables = var_config.get('variables')
                existing_variable_list = []
                for variable in variables:
                    if variable not in df.columns:
                        print("WARN: missing variable: {0} to calculate variable {1}".format(variable, var))
                    else:
                        existing_variable_list.append(variable)
                agg = var_config.get('agg', 'mean')
                divideby = var_config.get('divideby', 1)

                df[var] = df[existing_variable_list].agg(agg, axis=1) / divideby
                if var in post_processed_variables2:
                    post_processed_variables2.remove(var)
            post_processed_variables1 = post_processed_variables2
            num_new_variables = len(post_processed_variables1)

        return df

    def push_setpoints_to_db(self, cell, df):
        """
        push new setpoints to database to change FL setpoints
        :param cell: 'a' or 'b'
        :param df: dataframe with the following columns:
                    ['sup_air_flow_sp', 'sup_air_temp_sp', 'light_level_sp', 'battery_sp', 'zone_temp_hsp', 'zone_temp_csp']
                    sup_air_flow_sp: kg/s
                    sup_air_temp_sp: C
                    light_level_sp: 0-1
                    battery_sp: -3300 to 3300W
        :return: True if push is successful else False
        """

        if df.shape[0] > 1:
            print("Error: More than one row")
            return False
        if 'sup_air_flow_sp' in df.columns and 'sup_air_temp_sp' in df.columns:
            mass_flow_rate = df['sup_air_flow_sp'].values[0]
            temp = df['sup_air_temp_sp'].values[0]
            if mass_flow_rate == -1:
                df['sup_air_flow_sp_kgs'] = -1
                df['sup_air_flow_sp_cmh'] = -1
                df = df.drop(columns=['sup_air_flow_sp'])
            else:
                volume_flow_rate = self._convert_sup_air_flow_to_cmh_from_kgs(sup_air_flow_kgs=mass_flow_rate,
                                                                              sup_air_temp_C=temp)
                df['sup_air_flow_sp_kgs'] = mass_flow_rate
                df['sup_air_flow_sp_cmh'] = volume_flow_rate
                df = df.drop(columns=['sup_air_flow_sp'])
        df['time'] = datetime.datetime.utcnow()
        df = df.set_index('time')
        df_list = []
        for col in df.columns:
            df2 = df[[col]]
            df2.columns = ['value']
            df2['name'] = 'r'+cell + '_' + col+'_command'
            df_list.append(df2)
        final_df = pd.concat(df_list, axis=0)
        final_df = final_df.reset_index()

        query = 'insert into {0} values '.format(self.setpoint_table) + ','.join(final_df.apply(
            lambda x: "('{0}', '{1}', '{2}')".format(x['time'].strftime("%Y-%m-%d %H:%M:%S"), x['name'],
                                                          x['value']),
            axis=1).values) + ' on conflict (time, name) do update ' + 'SET value = excluded.value;'
        try:
            self.client.execute(query)
            print("pushed to table {0}".format(self.setpoint_table))
            return True
        except Exception as e:
            print("error occurred while pushing to table {0}, error={1}".format(self.setpoint_table, str(e)))
            return False

    def _get_density_kgm3(self, sup_air_temp_C):
        z = np.poly1d([-3.25360845e-08, 4.37156086e-05, -2.14726928e-02, 4.56125958e+00])
        sup_air_temp_K = sup_air_temp_C + 273.15
        return z(sup_air_temp_K)

    def _convert_sup_air_flow_to_cmh_from_kgs(self, sup_air_flow_kgs, sup_air_temp_C):
        density_kgm3 = self._get_density_kgm3(sup_air_temp_C=sup_air_temp_C)
        return sup_air_flow_kgs * 3600.0 / density_kgm3

