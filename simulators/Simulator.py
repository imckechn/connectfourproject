import jax
import time
import jax.numpy as jnp

from environment.connect_four import *
from config import default_config

class Simulator():
  '''simulates game to end from a state with given AIs'''

  def __init__(self, game_state, agents, key=None, config=default_config):
    self.agents = agents
    self.game_state = game_state
    self.config = config

    if key == None:
      self.key = jax.random.PRNGKey(int(time.time()))
    else:
      self.key = key

  def step(self, verbose=False):
    self.key, subkey = jax.random.split(self.key)
    choices = self.agents[self.game_state[3]&1].choose(self.game_state, subkey)
    self.game_state = play_move(self.game_state, choices.astype(jnp.uint64))

  def run(self, verbose=False):
    # games should only take 42 moves if all are legal.
    for i in jnp.arange(self.config['width'] * self.config['height']):

      if verbose:
        print(f'move # {i+1}')

      self.step()

    return get_winners(self.game_state, self.config)
