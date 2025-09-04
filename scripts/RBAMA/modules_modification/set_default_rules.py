from src.RBAMA import RBAMA
import argparse
from src.environments import registered_versions
import gymnasium as gym
import networkx as nx

def get_edge_number_as_index(agent):
    return agent.reasoning_unit.int_to_subscript(agent.reasoning_unit.G.number_of_edges()+1)

def set_default_rules(agent):

     # add fact nodes
    agent.reasoning_unit.G.add_node('D', type='morally relevant fact')
    agent.reasoning_unit.G.add_node('B', type='morally relevant fact')

    # add moral obligation nodes
    agent.reasoning_unit.G.add_node('R', type='moral obligation')
    agent.reasoning_unit.G.add_node('C', type='moral obligation')

    # add default rules with a hard-coded order among them
    agent.reasoning_unit.G.add_edge('D', 'R', lower_order={('B', 'C')}, name=f"δ{get_edge_number_as_index(agent)}")
    agent.reasoning_unit.G.add_edge('B', 'C', lower_order=set(), name=f"δ{get_edge_number_as_index(agent)}")

def main():
    parser = argparse.ArgumentParser(description="Manually set an agent's reasoning theory suitable to navigate the bridge environment.")

    parser.add_argument('agent_name', type=str, help='Name of the agent')

    args = parser.parse_args()

    agent, _ = RBAMA.setup_reasoning_agent(args.agent_name)
    agent.reasoning_unit.G = nx.DiGraph()

    set_default_rules(agent)

    RBAMA.save_agent(agent, args.agent_name)

if __name__ == '__main__':
    main()
