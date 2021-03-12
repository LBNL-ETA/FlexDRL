import argparse
import random
from drllib import trainer
random.seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG Training")
    parser.add_argument('--config_path', type=str, 
                        help='training hyperparameters configuration file path')
    
    args = parser.parse_args()
    train_ddpg = trainer.TrainDDPG(config_file = args.config_path)
    train_ddpg.training()