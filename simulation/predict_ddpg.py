import argparse
from drllib import predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG Prediction")
    parser.add_argument('--out_dir_path', type=str, 
                        help='directory where the results will be stored')
    parser.add_argument('--best_model_path', type=str, 
                        help='best model file path')
    parser.add_argument('--config_path', type=str, 
                        help='training hyperparameters configuration file path')
    
    args = parser.parse_args()
    pred_ddpg = predictor.PredDDPG(out_dir_path = args.out_dir_path, best_model = args.best_model_path, config_file = args.config_path)
    pred_ddpg.prediction()


# python simulation/predict_ddpg.py --out_dir_path "preds/" --best_model_path "saves_LRC/saves/DRL_Master/Test_203/best_-69177.359_7533385.dat" --config_path "simulation/test_configuration/Test_203_configuration.yaml"

# python simulation/predict_ddpg.py --out_dir_path "saves_LRC/saves/DRL_Master/" --best_model_path "saves_LRC/saves/DRL_Master/Test_203_8/best_-71747.316_7182995.dat" --config_path "simulation/test_configuration/Test_203_8_configuration.yaml"