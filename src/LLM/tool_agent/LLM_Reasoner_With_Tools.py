from src.LLM.baml_client import b
import gymnasium as gym
from src.environments import registered_versions
from src.environments.wrappers.cartesian_coordinates import Cartesian_Coordiantes
from enum import Enum
import numpy as np
from src.LLM.tool_agent import navigation_agent
import torch
from src.RBAMA.reasoning_unit import ReasoningUnit
import networkx as nx
from src.environments.bridge_random_instances import Random_Map_Layout
from src.utils import drl
import logging
import os
from src.RBAMA import reasoning_unit
from scripts.RBAMA.modules_modification.set_default_rules import set_default_rules
from src.LLM.tool_agent import navigation_agent
import logging

logger = logging.getLogger(__name__)

"""
methods for saving and loading an agent
"""
def get_agent_info(agent_name):
    dir_name = get_dir_name()
    file_path = os.path.join(dir_name, agent_name + ".pth")
    agent_info = torch.load(file_path)
    return agent_info

def load_modules(agent, agent_info):
    parameters = agent_info["navigation_agent_state_dict"]
    agent.navigation_agent.policy_dqn.load_state_dict(parameters)
    agent.navigation_agent.target_dqn.load_state_dict(parameters)
    agent.reasoning_unit = agent_info["reasoning_unit"]

def setup_reasoning_agent(agent_name):
    agent_info = get_agent_info(agent_name)
    env = agent_info["env"]
    agent = LMM_Reasoner_With_Tools(env)
    load_modules(agent, agent_info)
    return (agent, env)

def get_dir_name():
    current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
    return os.path.join(current_file_dir, 'agents')

def save_agent(agent, agent_name):
    
    save_dict = {
        "navigation_agent_state_dict": agent.navigation_agent.state_dict(),
        "env": agent.env,
        "reasoning_unit": agent.reasoning_unit
    }
    dir_name = get_dir_name()
    file_path= os.path.join(dir_name, agent_name + ".pth")
    logger.info(f"Agent saved to {file_path}")
    torch.save(save_dict, file_path)

"""class instantiating the DMM:
 an LLM as as central control unit and an RL agent that can be called as tool to navigate to a spot on the map
 also integrating the reasoning module
 """

class LMM_Reasoner_With_Tools:
    def __init__(self, env):
        self.env = env
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        self.navigation_agent = navigation_agent.RL_agent(env)

        self.reasoning_unit = reasoning_unit.ReasoningUnit(env, num_actions)

    def set_target_location(self, location):
        self.env.unwrapped.target_location = location

    def get_obligations(self, state):
        labels = self.env.get_lables()
        obligations = self.reasoning_unit.moral_obligations(labels, state)
        return obligations


"""helper functions"""
def set_target_location(env, location):
    env.unwrapped.target_location = location

def prepare_static_input(env):
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
    return water_tiles, goal_position

def extract_person_position(env):
    for person in env.persons:
        if env.bridge_map.in_water(person.position):
            position = [person.position[0], (env.width-1)-person.position[1]]
            return person, position
        
"""let the agent navigate the bridge environmnet and visualize its reasoning"""

def visualize(agent, env=None, state_reset=None, random_init="positions", seed=None):

        if not env:
            env = agent.env

        env.set_render_mode('human')
        env.unwrapped.target_location = np.array([0,0])  # reset target location
        agent.navigation_agent.policy_dqn.eval()  

        while True:
            if state_reset:
                state, _ = env.reset(state=state_reset)
            else:
                state, _ = env.reset(random_init=random_init, seed=seed)
            env.render()
            terminated = False     
            truncated = False         

            while(not terminated and not truncated):
                obligations = "No obligations"
                if "D" in env.get_lables():
                    person, drowning_position =extract_person_position(env)
                    obligations = f"Rescue the drowning person that needs to be rescued at position {person.slipping_spot}."
                #obligations = agent.get_obligations(state)
                args = {'target_position': goal_position}
                args['obligations'] = obligations
                agent_position={'row':env.get_agent_location()[0], 'column':(env.width-1)-env.get_agent_location()[1]}
                macroAction = b.MacroActionExecution(agent_position=agent_position, target_position=goal_position, obligations=obligations)
                print(macroAction)   
                if macroAction.tool_name == "navigation_agent":   
                    env.unwrapped.target_location = macroAction.goal_position_coordinates
                    with torch.no_grad():
                        action_preferences = [tensor.item() for tensor in navigation_agent.policy_dqn(navigation_agent.transformation(state))]
                        action = action_preferences.index(max(action_preferences))  
                state,_,terminated,truncated,_ = env.step(action)   

env = gym.make("bridge1_v1") # Use the Random_Map_Layout wrapper
env = Random_Map_Layout(env)
water_tiles, goal_position = prepare_static_input(env)
agent = LMM_Reasoner_With_Tools(env)

navigation_agent, _ = navigation_agent.setup_agent("navigation_agent_10000_episodes")
agent.navigation_agent = navigation_agent

reasoning_unit = reasoning_unit.ReasoningUnit(env, env.action_space.n)
set_default_rules(agent)

visualize(agent, state_reset=[17, 49, 49, 49, 41])

