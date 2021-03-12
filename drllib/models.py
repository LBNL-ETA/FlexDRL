import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from drllib import utils
from drllib.utils import BaseAgent

HID_SIZE = 128


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size, features1 = 400, features2 = 300):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, features1),
            nn.ReLU(),
            nn.Linear(features1, features2),
            nn.ReLU(),
            nn.Linear(features2, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size, features1 = 400, features2 = 300):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, features1),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(features1 + act_size, features2),
            nn.ReLU(),
            nn.Linear(features2, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class D4PGCritic(nn.Module):
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)

class AgentDDPG(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.4, ou_sigma=0.04, ou_epsilon=1.0, epsilon=0.3):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
        self.epsilon = epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = utils.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states
            actions += self.epsilon * np.random.normal(size=actions.shape)

        actions = np.clip(actions, -1, 1)

        return actions, new_a_states

# This class was modified to exclude the need of ptan
# class AgentD4PG(BaseAgent):
#     """
#     Agent implementing noisy agent
#     """
#     def __init__(self, net, device="cpu", epsilon=0.3):
#         self.net = net
#         self.device = device
#         self.epsilon = epsilon

#     def initial_state(self):
#         """
#         Should create initial empty state for the agent. It will be called for the start of the episode
#         :return: Anything agent want to remember
#         """
#         return None

#     def __call__(self, states, agent_states):
#         states_v = utils.float32_preprocessor(states).to(self.device)
#         mu_v = self.net(states_v)
#         actions = mu_v.data.cpu().numpy()
#         actions += self.epsilon * np.random.normal(size=actions.shape)
#         actions = np.clip(actions, -1, 1) #this prevent action to be outside of [-1,1]
#         return actions, agent_states


