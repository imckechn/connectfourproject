import haiku as hk
import jax, jax.numpy as jnp

import agents
import time

from environment.connect_four import *
from simulators.Simulator import Simulator

jnp.set_printoptions(linewidth=100000)

def forward(x):
    mlp = hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(7) # config['width']
    ])
    return mlp(x)

forward = hk.without_apply_rng(hk.transform(forward))

key = jax.random.PRNGKey(int(time.time()))

agent = agents.GuidedRandomAgent(forward, key)
agent2 = agents.RandomAgent()

key, subkey = jax.random.split(key)

state = init_game(100_000)
state = play_move(state, 0)
state = play_move(state, 0)
state = play_move(state, 0)
state = play_move(state, 0)
state = play_move(state, 0)
state = play_move(state, 0)

agent.choose(state, subkey)
