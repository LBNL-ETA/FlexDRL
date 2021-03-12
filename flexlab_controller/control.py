from flexlab_controller import controller
import datetime
import pytz
import time

if __name__ == "__main__":
    # BESTMODEL = "saves_LRC/saves/DRL_Master/Test_208/best_-70474.384_6307020.dat"
    BESTMODEL = "saves_LRC/saves/DRL_Master/Test_208_3/test_208_3_best_-85814.876_5956630.dat"
    EPRICE = "e_tariffs/e_d_price_2020_shed.csv"
    # number of features in the first second NN layers
    FEATURES1 = 400
    FEATURES2 = 400


    control = controller.FlexlabController(
        best_model_path = BESTMODEL,
        eprice_ahead = 3,
        eprice_path = EPRICE,
        device = "cpu",
        nn_features_1 = FEATURES1,
        nn_features_2 = FEATURES2
    )

    print("action_names: {0}".format(control.action_names))

    completed_minute = -1
    printed_minute = -1
    while True:
        t_now = datetime.datetime.now(pytz.timezone('US/Pacific'))
        minute_now = t_now.minute
        if minute_now % 15 == 0 and minute_now != completed_minute:
            try:
                t_now = datetime.datetime(t_now.year, t_now.month, t_now.day, t_now.hour, 15 * (t_now.minute // 15))
                print("t_now: {0}".format(t_now))
                actions = control.predict_action(t_now)

                actions['light_level_sp'] = 0

                if t_now.hour >= 7 and t_now.hour <20:
                    actions['light_level_sp'] = 1

                if t_now.hour >= 7 and t_now.hour < 19:
                    ## adding a standard difference to the command
                    actions['sup_air_temp_sp'] = actions['sup_air_temp_sp'] - 1.3

                print("actions at {0}".format(t_now))
                print(actions)

                print("pushing actions to FL interface")

                p_action = control.push_actions(actions=actions)

                print("push status= {0}".format(p_action))
                print(p_action)
                print("\n")
                completed_minute = minute_now
                printed_minute = minute_now
                time.sleep(1)
            except Exception as e:
                print("error occurred while running the controller. Error={0}".format(e))
        elif minute_now%1 == 0 and printed_minute!=minute_now:
            print("current time: {0}; waiting for the next 15th minute".format(t_now))
            printed_minute = minute_now
