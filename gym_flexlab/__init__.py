
from gym.envs.registration import register

register(
    id='FlexLab-v0',
    entry_point='gym_flexlab.envs:FlexLabEnv',
)
