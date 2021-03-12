import requests
import numpy as np
from driver import Driver, pd
import json
import logging
# TODO: setup logging

class Flexgrid(Driver):
    def __init__(self, config_file="read_data_config.yaml"):
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            super(Flexgrid, self).__init__(config_file=config_file)
        except Exception as e:
            self.logger.error("error reading config file={0} error={1}",format(config_file, str(e)))

        self.flexgrid_config = self.config.get('flexgrid')
        self.api_config = self.flexgrid_config.get('api')
        self.inverter_variable_map = self.flexgrid_config.get('inverter_variable_map')
        self.table_name = self.flexgrid_config.get('db_table_name', 'flexgrid')
        self.pv_scaling_divideby = self.flexgrid_config.get('pv_scaling_divideby', 2.8)

        self.host = self.api_config.get('host')
        self.port = self.api_config.get('port')

    def get_data(self):
        url = "{0}:{1}/read".format(self.host, self.port)
        try:
            response = requests.get(url)
            resp_content = response.content
            op = json.loads(resp_content)
        except Exception as e:
            self.logger.error("error querying API url={0} error={1}".format(url, str(e)))
            return {}

        time_now = self.get_utc_time_now()

        inverter_op_dictionary = {}

        for id_num in op:
            if id_num != '1' and id_num != '2' and id_num != '3':
                continue

            inverter_id = int(id_num)
            inverter_op = op[id_num]

            ac_charge = inverter_op.get("AC Charge", None)
            ac_frequeny = inverter_op.get("AC_F_Hz", None)
            ac_current = inverter_op.get("AC_I_A", None)
            ac_power_factor = inverter_op.get("AC_PF_%", None)
            ac_power = inverter_op.get("AC_P_W", None)
            ac_react = inverter_op.get("AC_Q_Var", None)
            ac_apparent = inverter_op.get("AC_S_VA", None)
            ac_coupling_voltage = inverter_op.get("AC_U-AB_V", None)
            active_power_control_enabled = inverter_op.get("Active Power Control", None)
            advanced_power_control_enabled = inverter_op.get("Advanced Power Control", None)
            battery_current = inverter_op.get("Batt_I_A", None)
            battery_power = inverter_op.get("Batt_P_W", None)
            battery_soc = inverter_op.get("Batt_SOC_1", None)
            battery_soh = inverter_op.get("Batt_SOH_1", None)
            battery_temperature = inverter_op.get("Batt_T_C", None)
            battery_voltage = inverter_op.get("Batt_U_V", None)
            battery_charge_limit = inverter_op.get("Charge Limit", None)
            battery_command_mode = inverter_op.get("Command Mode", None)
            battery_command_timeout = inverter_op.get("Command Timeout", None)
            battery_control_mode = inverter_op.get("Control Mode", None)
            dc_current = inverter_op.get("DC_I_A", None)
            dc_power = inverter_op.get("DC_P_W", None)
            dc_voltage = inverter_op.get("DC_U_V", None)
            inverter_default_mode = inverter_op.get("Default Mode", None)
            dc_discharge_limit = inverter_op.get("Discharge Limit", None)
            inverter_status = inverter_op.get("Inv_Stat_1", None)
            inverter_temperature = inverter_op.get("Inv_T_C", None)
            inverter_power_factor_control = inverter_op.get("Power Factor Control", None)
            inverter_rrcr_state = inverter_op.get("RRCR State", None)
            inverter_reactive_power_control = inverter_op.get("Reactive Power Control", None)
            inverter_reserved_capacity = inverter_op.get("Reserved Capacity", None)

            inverter_op_dictionary[inverter_id] = {
                'ac_charge': ac_charge,
                'ac_frequeny': ac_frequeny,
                'ac_current': ac_current,
                'ac_power_factor': ac_power_factor,
                'ac_power': ac_power,
                'ac_react': ac_react,
                'ac_apparent': ac_apparent,
                'ac_coupling_voltage': ac_coupling_voltage,
                'active_power_control_enabled': active_power_control_enabled,
                'advanced_power_control_enabled': advanced_power_control_enabled,
                'battery_current': battery_current,
                'battery_power': battery_power,
                'battery_soc': battery_soc,
                'battery_soh': battery_soh,
                'battery_temperature': battery_temperature,
                'battery_voltage': battery_voltage,
                'battery_charge_limit': battery_charge_limit,
                'battery_command_mode': battery_command_mode,
                'battery_command_timeout': battery_command_timeout,
                'battery_control_mode': battery_control_mode,
                'dc_current': dc_current,
                'dc_power': dc_power,
                'dc_voltage': dc_voltage,
                'inverter_default_mode': inverter_default_mode,
                'dc_discharge_limit': dc_discharge_limit,
                'inverter_status': inverter_status,
                'inverter_temperature': inverter_temperature,
                'inverter_power_factor_control': inverter_power_factor_control,
                'inverter_rrcr_state': inverter_rrcr_state,
                'inverter_reactive_power_control': inverter_reactive_power_control,
                'inverter_reserved_capacity': inverter_reserved_capacity
            }
        inverter_op_dictionary['time'] = time_now

        return inverter_op_dictionary

    def write_to_db(self):
        data = self.get_data()

        df_dict = {}
        time_now = data.get('time')
        df_dict['time'] = time_now
        for inverter_id in self.inverter_variable_map:
            variable_map = self.inverter_variable_map[inverter_id]
            output = data.get(inverter_id)
            for variable in variable_map:
                name = variable_map[variable]
                value = output.get(variable)
                if name == 'pv_generation_W':
                    value = max(0, value)
                    df_dict['pv_generation_scaled_W'] = value/self.pv_scaling_divideby
                elif name == 'ra_battery_rate_W' or name == 'rb_battery_rate_W':
                    value = -1* value
                df_dict[name] = [value]

        if len(df_dict) == 0:
            print("nothing to push to {0}".format(self.table_name))
        else:
            df = pd.DataFrame(df_dict)
            df = df.set_index('time')
            self.push_to_db(df=df, table_name=self.table_name)

if __name__ == "__main__":
    obj = Flexgrid(config_file='read_data_config.yaml')
    obj.write_to_db()
