from environment.connect_four import *
from config import default_config

import agents

import time
import jax, jax.numpy as jnp

class PlayerAgent(agents.Agent):
  def __init__(self, config=default_config):
    super().__init__(config=config)

  def choose(self, state, key=None, verbose=False):
    '''chooses actions using the state'''
    # in: state - a 4-tuple with the current game state (position, mask, active, turn)
    #     key - the jax random key
    # out: an action to take for each game

    draw_game(state)
    print('Which column would you like to play in')
    raw_input = input()

    choice = jnp.clip(int(raw_input), 0, self.config['width'])

    return choice[..., None]
