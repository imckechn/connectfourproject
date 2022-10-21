import jax, jax.numpy as jnp
import agents
from environment.connect_four import *
import math
import time

key = jax.random.PRNGKey(int(time.time()))
jnp.set_printoptions(precision=2, linewidth=100000)

agent = agents.UCBRolloutAgent(10, math.sqrt(2), 0.05, 1000)
game = init_game(10)

print(jnp.squeeze(agent.choose(game, key, verbose=True)))