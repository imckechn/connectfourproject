from simulators.TrainExpertSimulator import TrainExpertSimulator
from simulators.Simulator import Simulator
from environment.connect_four import *
import jax, jax.numpy as jnp

import math
import agents
import time

import pickle

import haiku as hk

from config import default_config

key = jax.random.PRNGKey(int(time.time()))

config = default_config

def model(x):
    return hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(config['width'])
    ])(x)

model = hk.without_apply_rng(hk.transform(model))

params = pickle.load(open("params.p", "rb"))

key, subkey, subkey1, subkey2 = jax.random.split(key, 4)

# make this regular rollouts for testing
rollout_agent = agents.GuidedRandomAgent(model, params, temperature=0.01)
agent = agents.UCBRolloutExpertAgent(14, math.sqrt(2), rollout_agent, batch_size=100)

game_state = init_game(100)
sim = TrainExpertSimulator(50, 1, game_state, agent, key=subkey)
new_params = sim.train(True)

pickle.dump(new_params, open("params.p", "wb"))