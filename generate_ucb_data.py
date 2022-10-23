from environment.connect_four import *
import agents
from simulators.Simulator import Simulator
import time

import jax

import pandas as pd

import math

# TODO: include variance for error bars
# include expert, train using supervised learning (save position, num times each arm chosen, and game outcome)
# create function that tried to match the frequency an arm is chosen and what it thinks the game outcome will be 

def graph_ucb_temperature(n_games, time, batch_size, confidence_level, temperatures, opponent, key, config=default_config):
    data = {'temperature': [], 'score': [], 'sem': []}

    for temp in temperatures:
        print(f'temperature: {temp}')
        key, subkey = jax.random.split(key)
        sim = Simulator(init_game(n_games), agents=[agents.UCBRolloutAgent(time, confidence_level, temp, batch_size, config), opponent], key=subkey, config=config)
        results = sim.run(verbose=True)
        score = jnp.mean(results)
        data['temperature'].append(temp)
        data['score'].append(score.item())
        data['sem'].append((jnp.std(results) / jnp.sqrt(n_games)).item())

    return pd.DataFrame(data)

'''
def graph_ucb_confidence_level(n_games, time, batch_size, confidence_levels=[0, 0.5, 1.0, math.sqrt(2), 1.5, 2.0, 2.5], opponent=agents.RolloutAgent(batch_size=100), config=default_config, key=None):
    data = {'confidence_levels': [], 'score': [], 'sem': []}

    if key == None:
        key = jax.random.PRNGKey(int(time.time()))

    # uses the same key for each simulation.
    for cl in confidence_levels:
        print(f'confidence_level: {cl} ')
        key, subkey = jax.random.split(key)
        sim = Simulator(init_game(n_games), agents=[agents.UCBRolloutAgent(time, cl, batch_size=batch_size, config=config), opponent], key = subkey, config=config)
        results = sim.run()
        score = jnp.mean(results)
        sem = jnp.std(results)/jnp.sqrt(n_games)
        data['confidence_levels'].append(cl)
        data['score'].append(score.item())
        data['sem'].append(sem.item())

    return pd.DataFrame(data)

def graph_ucb_confidence_level_p2(n_games, time, batch_size, confidence_levels=[0, 0.5, 1.0, math.sqrt(2), 1.5, 2.0, 2.5], opponent=agents.RolloutAgent(batch_size=100), config=default_config, key=None):
    data = {'confidence_levels': [], 'score': [], 'sem': []}

    if key == None:
        key = jax.random.PRNGKey(int(time.time()))

    # uses the same key for each simulation.
    for cl in confidence_levels:
        print(f'confidence_level: {cl} ')
        key, subkey = jax.random.split(key)
        sim = Simulator(init_game(n_games), agents=[opponent, agents.UCBRolloutAgent(time, cl, batch_size=batch_size, config=config)], key = subkey, config=config)
        results = sim.run()
        score = jnp.mean(results)
        sem = jnp.std(results)/jnp.sqrt(n_games)
        data['confidence_levels'].append(cl)
        data['score'].append(score.item())
        data['sem'].append(sem.item())

    return pd.DataFrame(data)


key = jax.random.PRNGKey(int(time.time()))
settings = [(100, 7), (200, 7), (300, 7), (400, 7), (500, 7), (600, 7), (700, 7)]
opponent_settings = [100, 200, 300, 400, 500, 600, 700]
file_name = './data/ucb/ro'
for i in range(len(settings)):
    print(f'STEP: {i}')
    key,subkey = jax.random.split(key)
    data = graph_ucb_confidence_level(500, *(settings[i]), opponent=agents.RolloutAgent(batch_size=opponent_settings[i]), key = subkey)
    data.to_pickle(f'{file_name}{i}')

key = jax.random.PRNGKey(int(time.time()))
settings = [(100, 7), (200, 7), (300, 7), (400, 7), (500, 7), (600, 7), (700, 7)]
opponent_settings = [100, 200, 300, 400, 500, 600, 700]
file_name = './data/ucb/2ro'
for i in range(len(settings)):
    print(f'STEP: {i}')
    key,subkey = jax.random.split(key)
    data = graph_ucb_confidence_level_p2(500, *(settings[i]), opponent=agents.RolloutAgent(batch_size=opponent_settings[i]), key = subkey)
    data.to_pickle(f'{file_name}{i}')
'''

key = jax.random.PRNGKey(int(time.time()))
data = graph_ucb_temperature(1000, 50, 10, math.sqrt(2), temperatures=[0.0000000001, 0.2], opponent=agents.RolloutAgent(batch_size=100), key=key)
data.to_pickle('./data/ucb2/temperatureplot')