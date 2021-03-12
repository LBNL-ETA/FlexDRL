from driver import Driver, pd, requests, np, datetime
import logging


class Weather(Driver):
    def __init__(self, config_file="read_data_config.yaml"):
        logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            super(Weather, self).__init__(config_file=config_file)
        except Exception as e:
            self.logger.error("error reading config file={0} error={1}",format(config_file, str(e)))

        self.weather_config = self.config.get('weather')
        self.api_config = self.weather_config .get('api')
        self.point_uuid_map = self.weather_config .get('point_uuid_map')
        self.cws_point_map = self.weather_config.get('cws_point_map')
        self.table_name = self.weather_config .get('db_table_name', 'weather')

        self.url = self.api_config.get('url')
        self.time_threshold = self.api_config.get('time_threshold', 60)
        self.flex_user = self.api_config.get('flex_user', 'weather')
        self.flex_password = self.api_config.get('flex_password')
        self.flexq_login = self.api_config.get('flexq_login', 'query')

    def write_to_db(self):
        time_now = self.get_utc_time_now()
        cws_point_list = self.get_data_cws(flexq_login=self.flexq_login, fl_username=self.flex_user,
                                           fl_password=self.flex_password)

        df = self.filter_cws_data(cws_point_list=cws_point_list, point_map=self.cws_point_map, time_now=time_now)

        if not df.empty:
            self.push_to_db(df, self.table_name)
        else:
            print("nothing to push to {0}".format(self.table_name))

if __name__ == "__main__":
    obj = Weather(config_file='read_data_config.yaml')
    obj.write_to_db()