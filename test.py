import jax, jax.numpy as jnp
import time

import math
from environment.connect_four import *

import agents
from simulators.Simulator import Simulator

agent1 = agents.UCBRolloutAgent(time=100, confidence_level=math.sqrt(2), temperature=0.05, batch_size=7)
agent2 = agents.RolloutAgent(batch_size=100)

key = jax.random.PRNGKey(int(time.time()))
sim = Simulator(init_game(100), [agent1, agent2], key)

print(jnp.mean(sim.run(verbose=True)))