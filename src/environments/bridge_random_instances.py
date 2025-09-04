import gymnasium as gym
import numpy as np
from gymnasium import spaces

"""wrapper for randomized drowning time"""
class Random_Map_Layout(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.size = self.bridge_map.width * self.bridge_map.height
            len_obs_dict = len(self.get_obs_dict())
            total_grid_cells = self.bridge_map.width  * self.bridge_map.height
            observation_size = len_obs_dict * total_grid_cells
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(observation_size,),
                dtype=np.float32
            )

        def random_target_pos(self):
            # choose the agent's location randomly 
            self.unwrapped.target_location[0] = self.np_random.integers(0, self.bridge_map.width)
            self.unwrapped.target_location[1] = self.np_random.integers(0, self.bridge_map.height)
            target_grid_type = self.bridge_map.get_grid_type(self.unwrapped.target_location)

            # ensure that the agent doesn't spawn in the water or at the goal position
            while target_grid_type == self.bridge_map.grid_types["water"] or np.equal(self.target_location, self.get_agent_location).all():
                self.unwrapped.target_location[0] = self.np_random.integers(0, self.bridge_map.width)
                self.unwrapped.target_location[1] = self.np_random.integers(0, self.bridge_map.height)
                target_grid_type = self.bridge_map.get_grid_type(self.unwrapped.target_location)

        def reset(self, seed=None, options=None, state = None, random_init= "no randomness"):
            self.random_target_pos()
            observation, info = self.env.reset(seed=seed, options=options, state = state, random_init= random_init)
            observation = self.observation(observation)
            return observation, info
        
        def get_obs_dict(self):
            obs_dict = self.env.get_obs_dict()

            target_window = np.zeros((self.bridge_map.height, self.bridge_map.width))
            target_x, target_y = self.target_location
            target_window[target_y, target_x] = 1

            del obs_dict['person_window']

            obs_dict["target_window"] = target_window

            return obs_dict

        def observation(self, observation):
            obs_dict = self.get_obs_dict()

            # flatten 
            agent_flat = obs_dict["agent_window"].flatten()
            water_flat = obs_dict["water_window"].flatten()
            target_flat = obs_dict["target_window"].flatten()

            # concatenate into one 1D array
            nn_input = np.concatenate([agent_flat, water_flat, target_flat])

            return np.array(nn_input.astype(np.float32))