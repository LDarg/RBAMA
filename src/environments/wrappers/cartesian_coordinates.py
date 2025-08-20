import gymnasium as gym
import numpy as np

class Cartesian_Coordiantes(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)

        def get_obs_dict(self):
            observation = {
                "agent_position": {"row": (self.height-1)-self.get_agent_location()[1], "column": self.get_agent_location()[0]},
                "person_positions": [{"person_position": list(person.position)} for person in self.persons]
            }
            return observation
        
        def observation(self, observation):
             return self.get_obs_dict()
        
        def reset(self, seed=None, options=None, state = None, random_init= "no randomness"):
            observation, info = self.env.reset(seed=seed, options=options, state = state, random_init=random_init)
            observation = self.observation(observation)
            return observation, info
        
