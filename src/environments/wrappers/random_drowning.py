from src.environments import registration
import gymnasium as gym
from collections import OrderedDict
import numpy as np
from gymnasium import spaces
from src.environments.bridge_person import Drowning_Random

class Random_Map_Layout(gym.Wrapper):
    def __init__(self, env, random_gen):
        super().__init__(env)
        self.random_gen = random_gen

    def set_random_target_location(self):
        self.unwrapped.target_location[0] = self.np_random.integers(0, self.bridge_map.width)
        self.unwrapped.target_location[1] = self.np_random.integers(0, self.bridge_map.height)
    
    def set_target_location(self, location):
        self.unwrapped.target_location = location

    def reset(self, seed=None, options=None, state=None, random_init="no randomness"):
        observation, info = self.env.reset(seed=seed, options=options, state=state, random_init=random_init)
        self.set_random_target_location(self.random_gen)
        return observation, info

"""wrapper for randomized drowning time"""
class Random_Drowning(gym.Wrapper):
        def __init__(self, env, prob):
            super().__init__(env)
            self.unwrapped.persons = []
            self.drowning_behavior = Drowning_Random(prob)
            self.initialize_persons(self.drowning_behavior)

        def reset(self, seed=None, options=None, state = None, random_init= "no randomness"):
            observation, info = self.env.reset(seed=seed, options=options, state = state, random_init= random_init)
            return observation, info
  
    