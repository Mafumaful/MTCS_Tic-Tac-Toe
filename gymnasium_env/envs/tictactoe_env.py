import gymnasium
from gymnasium import error, spaces, utils
from gym.utils import seeding
from pygame import *

class TicTacEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super().__init__()
    
    def step(self, action):
        pass
    
    def reset(self):
        pass
    
    def render(self, mode='human'):
        # render the environment to the screen
        pass
    
    def close(self):
        return super().close()