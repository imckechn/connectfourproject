from simulators.Simulator import Simulator
from environment.connect_four import *
import jax, jax.numpy as jnp

import agents
import time

import pickle

import haiku as hk

from config import default_config as config

def model(x):
    return hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(config['width'])
    ])(x)

model = hk.without_apply_rng(hk.transform(model))

params = pickle.load(open("params.p", "rb"))

guide = agents.GuidedRandomAgent(model, params, temperature=1)
agent1 = agents.GuidedRolloutAgent(guide)
agent2 = agents.RolloutAgent()

key = jax.random.PRNGKey(int(time.time()))
sim = Simulator(init_game(100), [agent1, agent2], key=key)
print(jnp.mean(sim.run(verbose=True)))