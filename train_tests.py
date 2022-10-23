from simulators.TrainExpertSimulator import TrainExpertSimulator
from environment.connect_four import *
import jax, jax.numpy as jnp

import math
import agents


key = jax.random.PRNGKey(42)

agent1 = agents.UCBRolloutExpertAgent(10, math.sqrt(2), key, temperature=0.001, batch_size=100)
agent2= agents.RolloutAgent(batch_size=100)

sim = TrainExpertSimulator(100, init_game(10), agent1, agent2)
sim.run()