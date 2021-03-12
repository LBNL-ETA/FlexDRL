import requests
from driver import Driver, pd
import json
import logging

class Lights(Driver):
    def __init__(self, config_file="read_data_config.yaml"):
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            super(Lights, self).__init__(config_file=config_file)
        except Exception as e:
            self.logger.error("error reading config file={0} error={1}",format(config_file, str(e)))

        self.lights_config = self.config.get('lights')
        self.api_config = self.lights_config .get('api')
        self.cree_device_uuid_map = self.lights_config .get('cree_device_uuid_map')
        self.table_name = self.lights_config .get('db_table_name', 'lights')

        self.host = self.api_config.get('host')
        self.port = self.api_config.get('port')
        self.app_id = self.api_config.get('app_id')
        self.api_key = self.api_config.get('api_key')

    def get_cree_data(self, time_now):
        url = "{0}/smartcast-api/v1/devices?page_size=200&page_number=1".format(self.host)

        print (url)
        headers = {
            'accept': 'application/json',
            'APPID': self.app_id,
            'APIKEY': self.api_key
        }

        op = {}
        try:
            response = requests.get(url=url, headers=headers, verify=False)
            print(response)
            resp_content = response.content
            op = json.loads(resp_content)
        except Exception as e:
            self.logger.error("error querying API url={0} error={1}".format(url, str(e)))

        op_devices = op.get('devices', [])

        data = {}
        data['time'] = [time_now]
        for device_dict in op_devices:
            op_uuid = device_dict.get('uuid')
            light_id = self.cree_device_uuid_map.get('ra', {}).get(op_uuid, None)
            if light_id == None:
                light_id = self.cree_device_uuid_map.get('rb', {}).get(op_uuid, None)
                if light_id == None:
                    print("can't find {0} in cree_device_uuid_map".format(op_uuid))
                    continue
                cell = 'b'
            else:
                cell = 'a'
            sensor_data = device_dict.get('sensor_data')
            light_level = sensor_data.get('light_level')
            ambient_light_level = sensor_data.get('ambient_light_level')
            if ambient_light_level != 65535:
                data['r{0}_ambient{1}'.format(cell, light_id)] = [ambient_light_level]

            data['r{0}_level{1}'.format(cell, light_id)] = [light_level]

        return data

    def write_to_db(self):
        time_now = self.get_utc_time_now()
        cree_data = self.get_cree_data(time_now=time_now)

        if len(cree_data) != 0:
            df = pd.DataFrame(cree_data)
            df = df.set_index('time')

            self.push_to_db(df, self.table_name)
        else:
            print("nothing to push to {0}".format(self.table_name))

if __name__ == "__main__":
    obj = Lights(config_file='read_data_config.yaml')
    obj.write_to_db()
