from environment.connect_four import *
import agents
from simulators.Simulator import Simulator
import time

import jax

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

def graph_rollout_parameter(shape, rollouts=[1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], opponent=agents.RolloutAgent(n_rollouts=100), config=default_config, key=None):
    data = {'rollouts':[], 'score':[]}

    if key == None:
        key = jax.random.PRNGKey(int(time.time()))

    # uses the same key for each simulation.
    for r in rollouts:
        print(f'starting n_rollouts = {r} ')
        sim = Simulator(init_game(shape), agents=[agents.RolloutAgent(n_rollouts=r), opponent], key = key, config=config)
        score = jnp.mean(sim.run())
        data['rollouts'].append(r)
        data['score'].append(score.item())
        print(f'done n_rollouts = {r}')

    return pd.DataFrame(data)

def graph_rollout_parameter_as_second_player(shape, rollouts=[1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000], opponent=agents.RolloutAgent(n_rollouts=100), config=default_config, key=None):
    data = {'rollouts':[], 'score':[]}

    if key == None:
        key = jax.random.PRNGKey(int(time.time()))

    # uses the same key for each simulation.
    for r in rollouts:
        print(f'starting n_rollouts = {r} ')
        sim = Simulator(init_game(shape), agents=[opponent, agents.RolloutAgent(n_rollouts=r)], key = key, config=config)
        score = jnp.mean(sim.run())
        data['rollouts'].append(r)
        data['score'].append(score.item())
        print(f'done n_rollouts = {r}')

    return pd.DataFrame(data)

'''
key = jax.random.PRNGKey(int(time.time()))

fname= './data/vr100.pkl'
data = graph_rollout_parameter(300, opponent=agents.RolloutAgent(n_rollouts=100), key=key)
data.to_pickle(fname)

fname= './data/vr200.pkl'
data = graph_rollout_parameter(300, opponent=agents.RolloutAgent(n_rollouts=400), key=key)
data.to_pickle(fname)

fname= './data/vr300.pkl'
data = graph_rollout_parameter(300, opponent=agents.RolloutAgent(n_rollouts=300), key=key)
data.to_pickle(fname)

fname= './data/vr400.pkl'
data = graph_rollout_parameter(300, opponent=agents.RolloutAgent(n_rollouts=400), key=key)
data.to_pickle(fname)

fname= './data/vr500.pkl'
data = graph_rollout_parameter(300, opponent=agents.RolloutAgent(n_rollouts=500), key=key)
data.to_pickle(fname)

fname= './data/vr600.pkl'
data = graph_rollout_parameter(300, opponent=agents.RolloutAgent(n_rollouts=600), key=key)
data.to_pickle(fname)

fname= './data/vr700.pkl'
data = graph_rollout_parameter(300, opponent=agents.RolloutAgent(n_rollouts=700), key=key)
data.to_pickle(fname)
'''

key = jax.random.PRNGKey(int(time.time()))

fname= './data/2vr100.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RolloutAgent(n_rollouts=100), key=key)
data.to_pickle(fname)

fname= './data/2vr200.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RolloutAgent(n_rollouts=400), key=key)
data.to_pickle(fname)

fname= './data/2vr300.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RolloutAgent(n_rollouts=300), key=key)
data.to_pickle(fname)

fname= './data/2vr400.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RolloutAgent(n_rollouts=400), key=key)
data.to_pickle(fname)

fname= './data/2vr500.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RolloutAgent(n_rollouts=500), key=key)
data.to_pickle(fname)

fname= './data/2vr600.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RolloutAgent(n_rollouts=600), key=key)
data.to_pickle(fname)

fname= './data/2vr700.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RolloutAgent(n_rollouts=700), key=key)
data.to_pickle(fname)

fname= './data/2vrp.pkl'
data = graph_rollout_parameter_as_second_player(300, opponent=agents.RandomPlusAgent(), key=key)
data.to_pickle(fname)

