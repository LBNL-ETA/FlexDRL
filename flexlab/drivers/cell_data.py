from driver import Driver, pd, requests, np, datetime
import logging
from thermal_power_calculations import calculater_water_energy_flow_rate


class Cell_Data(Driver):
    def __init__(self, config_file="read_data_config.yaml"):
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            super(Cell_Data, self).__init__(config_file=config_file)
        except Exception as e:
            self.logger.error("error reading config file={0} error={1}",format(config_file, str(e)))

        self.cell_data_config = self.config.get('cell_data')
        self.api_config = self.cell_data_config .get('api')
        self.flex_user = self.api_config.get('flex_user', 'btmder')
        self.flex_password = self.api_config.get('flex_password')
        self.flexq_login = self.api_config.get('flexq_login', 'queryxr')

        self.cella_config = self.cell_data_config.get('cella')
        self.cellb_config = self.cell_data_config.get('cellb')
        self.power_config = self.cell_data_config.get('power')

        self.cella_table_name = self.cella_config.get('db_table_name', 'cella')
        self.cella_cws_point_map = self.cella_config.get('cws_point_map')

        self.cellb_table_name = self.cellb_config.get('db_table_name', 'cellb')
        self.cellb_cws_point_map = self.cellb_config.get('cws_point_map')

        self.power_table_name = self.power_config.get('db_table_name', 'power')
        self.power_meter_channels_map = self.power_config.get('power_meter_channels_map')
        self.power_thermal_channels_map = self.power_config.get('power_thermal_channels_map')

    def get_electric_load(self, cws_point_list, time_now):
        useful_points = {}
        for name in self.power_meter_channels_map:
            channels = self.power_meter_channels_map[name]
            for c in channels:
                cws_channel = "PR{0}-ActivePower{1}".format(c[0], c[1:])
                useful_points[cws_channel] = c

        df = self.filter_cws_data(cws_point_list=cws_point_list, point_map=useful_points, time_now=time_now)

        if df.empty:
            return pd.DataFrame()

        final_data = {}
        final_data['time'] = [time_now]
        for var in self.power_meter_channels_map:
            columns = self.power_meter_channels_map[var]
            final_data[var] = [df[columns].sum(axis=1).values[0]]

        final_df = pd.DataFrame(final_data)
        final_df = final_df.set_index('time')

        return final_df

    def get_thermal_load(self, cws_point_list, power_electric_df, time_now):
        useful_points = {}
        for name in self.power_thermal_channels_map:
            if name != 'ra_hvac_th_load_W' and name != 'rb_hvac_th_load_W':
                variable_map = self.power_thermal_channels_map[name]
                useful_points.update(variable_map)

        df = self.filter_cws_data(cws_point_list=cws_point_list, point_map=useful_points,
                                 time_now=datetime.datetime.utcnow())
        if df.empty:
            return pd.DataFrame()

        useful_values = df.to_dict(orient='records')[0]

        final_data = {}
        final_data['time'] = [time_now]

        for point in self.power_thermal_channels_map:
            if point != 'ra_hvac_th_load_W' and point != 'rb_hvac_th_load_W':
                split_point = point.split('_')
                cell = split_point[0]
                hw_or_chw = split_point[1]
                flow_LPM = useful_values['{0}_{1}_flow'.format(cell, hw_or_chw)]
                sup_temp_C = useful_values['{0}_{1}_sup_temp'.format(cell, hw_or_chw)]
                ret_temp_C = useful_values['{0}_{1}_ret_temp'.format(cell, hw_or_chw)]
                final_data[point] = [calculater_water_energy_flow_rate(water_flow_LPM=flow_LPM,
                                                                      water_sup_temp_C=sup_temp_C,
                                                                      water_ret_temp_C=ret_temp_C,
                                                                      hw_or_chw=hw_or_chw)]

        final_data['ra_hvac_th_load_W'] = [
            power_electric_df.loc[time_now, 'ra_fan_power_W'] + final_data['ra_hw_th_load_W'][0] + final_data[
                'ra_chw_th_load_W'][0]]
        final_data['rb_hvac_th_load_W'] = [
            power_electric_df.loc[time_now, 'rb_fan_power_W'] + final_data['rb_hw_th_load_W'][0] + final_data[
                'rb_chw_th_load_W'][0]]

        final_df = pd.DataFrame(final_data)
        final_df = final_df.set_index('time')

        return final_df


    def write_to_db(self):
        time_now = self.get_utc_time_now()
        cws_point_list = self.get_data_cws(flexq_login=self.flexq_login, fl_username=self.flex_user,
                                           fl_password=self.flex_password)

        cella_df = self.filter_cws_data(cws_point_list=cws_point_list, point_map=self.cella_cws_point_map,
                                        time_now=time_now)

        cellb_df = self.filter_cws_data(cws_point_list=cws_point_list, point_map=self.cellb_cws_point_map,
                                        time_now=time_now)

        power_electric_df = self.get_electric_load(cws_point_list=cws_point_list, time_now=time_now)

        power_thermal_df = self.get_thermal_load(cws_point_list=cws_point_list, power_electric_df=power_electric_df,
                                                 time_now=time_now)

        if not cella_df.empty:
            self.push_to_db(cella_df, self.cella_table_name)
        else:
            print("nothing to push to {0}".format(self.cella_table_name))

        if not cellb_df.empty:
            self.push_to_db(cellb_df, self.cellb_table_name)
        else:
            print("nothing to push to {0}".format(self.cellb_table_name))

        if not power_electric_df.empty:
            self.push_to_db(power_electric_df, self.power_table_name)
        else:
            print("no electric load data pushed to {0}".format(self.power_table_name))

        if not power_thermal_df.empty:
            self.push_to_db(power_thermal_df, self.power_table_name)
            print("pushed thermal power to {0}".format(self.power_table_name))
        else:
            print("no thermal load data pushed to {0}".format(self.power_table_name))

if __name__ == "__main__":
    obj = Cell_Data(config_file='read_data_config.yaml')
    obj.write_to_db()