
from gym_flexlab.envs import flexlab_env
#import sys

#sys.path.append("//Users/mkiran/SWProjects/calibers/DRL_FLEXLAB/")

import os
import math
import time
import gym
import argparse
from tensorboardX import SummaryWriter

#import roboschool

from drllib import tpro_model, trpo, utils, kfac


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


#ENV_ID = "RoboschoolHalfCheetah-v1"


GAMMA = 0.99
REWARD_STEPS = 5
BATCH_SIZE = 32
LEARNING_RATE_ACTOR = 1e-3
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-3
ENVS_COUNT = 16

TEST_ITERS = 20
CUDA = False
RunName = "Test70"

def test_net(net, env, writer, exp_idx, count=1, device="cpu"):
    # print(exp_idx)
    # shift = exp_idx + 50000
    
    rewards = 0.0
    e_costs = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        # t_idx = 0
        while True:
            # t_idx += 1
            #print("in true test")
            obs_v = utils.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            #print("action")
            #print(action)
            obs, reward, done, divers = env.step(action)
            #print("done:")
            #print(done)
            e_cost = divers[0]

            rewards += reward
            e_costs += e_cost
            steps += 1
            if done:
                break
    print("in test_net")
    print(rewards)
    return rewards / count, steps / count, e_costs / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    #parser.add_argument("-n", "--name", required=True, help="Name of the run")
    #parser.add_argument("-e", "--env", default=ENV_ID, help="Environment id, default=" + ENV_ID)
    args = parser.parse_args()
    device = torch.device("cuda" if CUDA else "cpu")


    advantage=[]
    values=[]
    loss_value=[]
    batch_rewards=[]
    loss_entropy=[]
    loss_total=[]
    loss_policy=[]
    test_reward=[]
    test_steps=[]



    save_path = os.path.join("saves", "acktr-" + RunName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    sim_days = 365


    env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v2/FlexlabXR_v2_SFO_2015.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2015.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2015.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2015,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 365,
                                 step_size = 900)


    test_env = flexlab_env.FlexLabEnv(envelope_path = 'fmu_models/EPlus_FMU_v2/FlexlabXR_v2_SFO_2017.fmu',
                                 battery_path = 'fmu_models/battery.fmu',
                                 pv_path =  'fmu_models/PV_FMU/PV_SFO_2017.fmu',
                                 eprice_path = 'e_tariffs/e_d_price_2017.csv', 
                                 chiller_COP = 3.0, 
                                 boiler_COP = 0.95,
                                 sim_year = 2017,
                                 tz_name = 'America/Los_Angeles',
                                 sim_days = 365,
                                 step_size = 900)
    
    #env = [gym.make(args.env) for _ in range(ENVS_COUNT)]
    #test_env = gym.make(args.env)
    
    #net_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.shape[0]).to(device)
    #net_crt = model.ModelCritic(envs[0].observation_space.shape[0]).to(device)
    #print(net_act)
    #print(net_crt)

    net_act = tpro_model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net_crt = tpro_model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-acktr_" + RunName)
    start_time=time.time()
    print("&&&&&&&&&&&&&&&&&&&&&action_space")
    print(env.action_space.shape[0])

    agent = tpro_model.AgentA2C(net_act, device=device)
    exp_source = utils.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)

    opt_act = kfac.KFACOptimizer(net_act, lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    batch = []
    best_reward = None
    with utils.RewardTracker(writer) as tracker:
        with utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                print("******************************step_idx")
                print(step_idx)
                rewards_steps = exp_source.pop_rewards_steps()
                print("reward_steps")
                print(rewards_steps)

                if rewards_steps:
                    print("new rewards")
                    print(rewards_steps)
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % ((env.n_steps -1) * TEST_ITERS) == 0:
                    print("in step loop")

                    ts = time.time()
                    rewards, steps, e_costs = test_net(net_act, test_env, writer, step_idx, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    test_reward.append(rewards)
                    test_steps.append(steps)

                    if best_reward is None or best_reward < rewards:
                        print("*********************In saving loop")
                        print(best_reward)

                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                                           
                            torch.save(net_act.state_dict(), fname)
                            dfadvvalueloss=pandas.DataFrame(data={"advantage": advantage, "values":values,"loss_values":loss_value})
                            dftestrew=pandas.DataFrame(data={"test_reward":test_reward,"test_step": test_steps})
                            dfadvvalueloss.to_csv("performancedata/dfadvvalueloss.csv",sep=',',index=False)   
                            dftestrew.to_csv("performancedata/dftestrew.csv",sep=',',index=False)   

                            dfnewaktr=pandas.DataFrame(data={"batch_rewards": batch_rewards, "loss_entropy":loss_entropy,"loss_total":loss_total,"loss_policy":loss_policy})
                            dfnewaktr.to_csv("performancedata/dfnewaktr.csv",sep=',',index=False)   
                       
                        
                        best_reward = rewards
                        print("new best rewards")
                        print(best_reward)

                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = \
                    common.unpack_batch_a2c(batch, net_crt, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                mu_v = net_act(states_v)
                log_prob_v = calc_logprob(mu_v, net_act.logstd, actions_v)
                if opt_act.steps % opt_act.Ts == 0:
                    opt_act.zero_grad()
                    pg_fisher_loss = -log_prob_v.mean()
                    opt_act.acc_stats = True
                    pg_fisher_loss.backward(retain_graph=True)
                    opt_act.acc_stats = False

                opt_act.zero_grad()
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                loss_policy_v = -(adv_v * log_prob_v).mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                writer.add_scalar("advantage", adv_v, step_idx)
                writer.add_scalar("values", value_v, step_idx)
                writer.add_scalar("batch_rewards", vals_ref_v, step_idx)
                writer.add_scalar("loss_entropy", entropy_loss_v, step_idx)
                writer.add_scalar("loss_policy", loss_policy_v, step_idx)
                writer.add_scalar("loss_value", loss_value_v, step_idx)
                writer.add_scalar("loss_total", loss_v, step_idx)
                advantage.append(adv_v)
                values.append(value_v)
                loss_value.append(loss_value_v)
                batch_rewards.append(vals_ref_v)
                loss_entropy.append(entropy_loss_v)
                loss_total.append(loss_v)
                loss_policy.append(loss_policy_v)

                print("###########################################End of main")
                print("step : %d" % step_idx)
