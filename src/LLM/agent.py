from baml_client import b
import gymnasium as gym
from src.environments import registered_versions
from src.environments.wrappers.cartesian_coordinates import Cartesian_Coordiantes
from enum import Enum
import numpy as np

env = gym.make("bridge1_v1")
env = Cartesian_Coordiantes(env)
water_tiles = {
    "positions": [
        {"column": water_tile[0], "row": (env.height-1)-water_tile[1]} # switch x and y
        for water_tile in env.water_tiles
    ]
}
goal_position = {
      "row": env.target_location[0],
      "column": (env.width-1)-env.target_location[1]
}

class Action(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

move_to_action = {
    (1, 0): 0,   # right
    (0, -1): 1,   # down
    (-1, 0): 2,  # left
    (0, 1): 3   # up
}

def action_index(new_pos, old_pos):
    delta= tuple(new_pos - old_pos)
    action_index = move_to_action.get(delta, None)  # convert to tuple for dict key
    return action_index

danger_information = lambda: "You are next to a water tile. Take care not to step into it." if env.next_to_water else "Nothing"


def visualize(state_reset=None, random_init= "no randomness"):

        env.set_render_mode('human')

        while True:
            if state_reset:
                state, _ = env.reset(state=state_reset)
            else:
                state, _ = env.reset(random_init=random_init)
            terminated = False     
            truncated = False                  

            while(not terminated and not truncated): 
                    test =water_tiles
                    test1 = state["agent_position"]
                    hint = "You are next to a water tile. Take care not to step into it." if env.bridge_map.next_to_water(env.get_agent_location()) else "Nothing"
                    action, reasoning = b.ChooseAction(agent_position=state["agent_position"], water_tiles=water_tiles, goal_position=goal_position, grid_size=7, hint=hint)
                    test2 = [state["agent_position"]["column"], state["agent_position"]["row"]]
                    action = action_index(np.array(action[1]), np.array([state["agent_position"]["column"], state["agent_position"]["row"]]))
                    #action = Action[str(action[1]).upper()].value
                    #action, reasoning = b.TestWaterTiles(agent_position=state["agent_position"], water_tiles=water_tiles, goal_position=goal_position)

                    state,_,terminated,truncated,_ = env.step(action)

visualize()
