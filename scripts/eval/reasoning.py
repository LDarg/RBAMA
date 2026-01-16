#!/usr/bin/env python3
from src.RBAMA import RBAMA
import argparse
import ast 

import torch
import networkx as nx
import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

"""
sets upt the window for the reason theory plot
"""
def setup_reason_theory_plot():

    #allows real time update for the plot
    plt.ion()

    plt.rcParams['toolbar'] = 'none'
    plt.rcParams['font.family'] = 'serif'

    #set the size and create the figure
    figsize = (6, 6)
    fig = plt.figure(figsize=figsize)

    n = 1000
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    _, Y = np.meshgrid(x, y)
    ax_bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    ax_bg.set_axis_off()

    #create two subplots: one where the graphical representation of the default rules is displayed and one that shows the order among them
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])  
    ax_order = fig.add_subplot(gs[0])  
    ax_theory = fig.add_subplot(gs[1]) 

    #disables displaying the axis
    ax_order.set_axis_off()
    ax_theory.set_axis_off()
    ax_bg.set_axis_off()

    # no padding around subplots (compact layout)
    plt.tight_layout(pad=0)
    fig.canvas.manager.set_window_title("Reason Theory")

    return (ax_order, ax_theory)

"""
renders an RBAMAs reason theory
"""
def render_graph_reasoning(agent, moral_obligations, ax_order, ax_theory, env, state):
    # clears the plot
    ax_order.cla()
    ax_theory.cla()

    # sets the node colors: red := the agent recognizes the moral olbigations as overall binding; blue: the agent does not recognize the moral oblgiation as overall binding
    node_colors = []
    for node, data in agent.reasoning_unit.G.nodes(data=True):
        if node in moral_obligations:
            node_colors.append('red')  
        elif node in env.get_lables():
            node_colors.append('red') 
        else:
            node_colors.append('lightblue')

    # Use node labels directly without modification
    node_labels = {node: node for node in agent.reasoning_unit.G.nodes()}

    # gets all propositions that are known to the agent as morally relevant facts
    statement_nodes = [node for node, data in agent.reasoning_unit.G.nodes(data=True) if data.get('type') == 'morally relevant fact']

    # sets graphical represenation of rules
    positions = nx.bipartite_layout(agent.reasoning_unit.G, nodes=statement_nodes, aspect_ratio=2, center=(0, 0), align='horizontal', scale=0.7)
    nx.draw(
        agent.reasoning_unit.G, pos=positions, labels=node_labels, with_labels=True, 
        node_size=2000, node_color=node_colors, edge_color='black', width=3, ax=ax_theory, 
        font_size=24, font_family='serif'
    )

    # add labels for default rules (including numeration) 
    edge_labels = nx.get_edge_attributes(agent.reasoning_unit.G, 'name')
    nx.draw_networkx_edge_labels(agent.reasoning_unit.G, pos=positions, edge_labels=edge_labels, 
                                 font_color='black', ax=ax_theory, font_size=16, rotate=False)

    # creates text field to display the order among the rules
    order_text = ""
    obligations = set()
    for rule in agent.reasoning_unit.G.edges(data=True):
        _, _, rule_data = rule
        lower_order = rule_data.get('lower_order', None)
        if lower_order:
            for other_rule in lower_order:
                other_rule_name = agent.reasoning_unit.G.get_edge_data(other_rule[0], other_rule[1]).get('name', 'No name attribute')
                rule_name = agent.reasoning_unit.G.get_edge_data(rule[0], rule[1]).get('name', 'No name attribute')
                order_text += f"{rule_name} > {other_rule_name} " 


    lables = env.get_lables()
    obligations = set()
    for rule in agent.reasoning_unit.G.edges(data=True):
        if rule[0] in lables:
            obligations.add(rule[1])
    if agent.reasoning_unit.conflicted_actions(obligations, state):
        ax_order.text(
            0.5, 0.3,  
            f"Conflicted!\n{order_text}", 
            fontsize=20,
            color='red',
            ha='center',
            va='center',
            fontweight='bold',
            transform=ax_order.transAxes
        )
    else:
        ax_order.text(
            0.5, 0.5,  
            order_text, 
            fontsize=24, 
            color='black', 
            ha='center', 
            va='center',
            fontweight='bold',
            transform=ax_order.transAxes  
        )   

    # adds a margin from the subplots to the figure window
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1) 

    # displaying the axes
    ax_order.set_axis_off()
    ax_theory.set_axis_off()

def visualize_reasoning(agent, env, reset_state=None, random_init="positions"):

        env.set_render_mode("human")
        env.metadata["render_fps"] = 3

        ax_order, ax_theory = setup_reason_theory_plot()

        agent.policy_dqn.eval()   

        while True:
            if reset_state is not None:
                state, _ = env.reset(state=reset_state)
            else:
                state, _ = env.reset(random_init=random_init)
            terminated = False     
            truncated = False                

            # execute moral policy
            while(not terminated and not truncated): 
                moral_obligations = agent.reasoning_unit.moral_obligations(env.get_lables(), state)
                render_graph_reasoning(agent,moral_obligations, ax_order, ax_theory, env, state)
                env.render()
                morally_permissible_actions = agent.filter_actions(env, state)
                # select morally permissible action  
                with torch.no_grad():
                    #take an action that is conform with the agent's moral obligations according to its current reason theory
                    action_preferences = [tensor.item() for tensor in agent.policy_dqn(agent.transformation(state))]
                    action = action_preferences.index(max(action_preferences))
                    choice_list = list(action_preferences)
                    while action not in morally_permissible_actions:
                        choice_list.remove(max(choice_list))
                        action = action_preferences.index(max(choice_list))

                state,_,terminated,truncated,_ = env.step(action)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Visualize the reasoning of the RBAMA during navigating the environmnet.")
   parser.add_argument('agent_name', type=str, help="Name of the agent")
   parser.add_argument('--state_reset', type=ast.literal_eval, help='"List of values values specifying the positions of the agent and each person on the flattened map, following the pattern: [agent_position, position_person_id_1, position_person_id_2, position_person_id_3, position_person_id_4]"')

   args = parser.parse_args()
   agent_name = args.agent_name
   agent, agent_training_env = RBAMA.setup_reasoning_agent(agent_name)
   state_reset = args.state_reset

   visualize_reasoning(agent, agent_training_env, reset_state=state_reset)

    