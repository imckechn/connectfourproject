from environment.connect_four import *
from simulators.Simulator import Simulator

import jax, jax.numpy as jnp
import haiku as hk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time

import pickle

from agents.UCBRolloutExpertAgent import UCBRolloutExpertAgent
from agents.RolloutAgent import RolloutAgent

def load_params_from_file(params_path):
    return pickle.load(open(params_path, 'rb'))

def load_UCB_agent_from_file(params_path, model):
    params = load_params_from_file(params_path)
    
    return UCBRolloutExpertAgent(100, model, params, 10)

def evaluate_expert_UCB(n_games, params_path0, params_path1, model, key):
    params0 = load_params_from_file(params_path0)
    params1 = load_params_from_file(params_path1)

    return evaluate_expert_UCB_params(n_games, params0, params1, model, key)

def evaluate_expert_UCB_params(n_games, params0, params1, model, key):
    
    agent0 = UCBRolloutExpertAgent(100, model, params0, 10)
    agent1 = UCBRolloutExpertAgent(100, model, params1, 10)

    return evaluate_agent_v_agent(n_games, agent0, agent1, key)

def evaluate_agent_v_agent(n_games, agent0, agent1, key):

    key, subkey = jax.random.split(key)
    start_time = time.perf_counter()
    sim = Simulator(init_game(n_games), [agent0, agent1], subkey)
    resultsA = sim.run(verbose=True)
    
    key, subkey = jax.random.split(key)
    sim = Simulator(init_game(n_games), [agent1, agent0], subkey)
    resultsB = sim.run(verbose=True)
    end_time = time.perf_counter()

    agent0_wins = jnp.count_nonzero(resultsA == -1) + jnp.count_nonzero(resultsB == 1)
    agent1_wins = jnp.count_nonzero(resultsA == 1) + jnp.count_nonzero(resultsB == -1)
    ties = jnp.count_nonzero(resultsA == 0) + jnp.count_nonzero(resultsB == 0)

    print(f'agent 0 total wins: {agent0_wins}')
    print(f'agent 1 total wins: {agent1_wins}')
    print(f'ties: {ties}')
    print(f'time elapsed: {end_time - start_time} seconds.')

    return agent0_wins, agent1_wins, ties

if __name__ == '__main__':
    param_paths = [
        ['./datasets/ucb_net_v9/dataset_25_params.pk', 'Generation 25'],
        ['./datasets/ucb_net_v9/dataset_39_params.pk', 'Generation 39']
    ]
    #v9 - 25 BEST SO FAR

    # define the model
    def model(x):
        return hk.Sequential([
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(7, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))
        ])(x)

    model = hk.without_apply_rng(hk.transform(model))

    key = jax.random.PRNGKey(int(time.time()))

    data = {'agent0_name': [], 'agent1_name': [], 'agent_0_wins': [], 'agent_1_wins': [], 'ties': []}

    for i in range(len(param_paths) - 1):
        agent0_wins, agent1_wins, ties = evaluate_expert_UCB(100, param_paths[i][0], param_paths[i+1][0], model, key)

        data['agent0_name'].append({param_paths[i][1]})
        data['agent1_name'].append({param_paths[i+1][1]})
        data['agent_0_wins'].append(float(agent0_wins))
        data['agent_1_wins'].append(float(agent1_wins))
        data['ties'].append(float(ties))

    data = pd.DataFrame(data)

    pickle.dump(data, open('compare_data2.pk', 'wb'))