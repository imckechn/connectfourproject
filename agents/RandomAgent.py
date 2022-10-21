from environment.connect_four import *
from config import default_config

import agents

import time
import jax

class RandomAgent(agents.Agent):
  def __init__(self, config=default_config):
    super().__init__(config=config)

  def choose(self, state, key=None):
    '''chooses actions using the state'''
    # in: state - a 4-tuple with the current game state (position, mask, active, turn)
    #     key - the jax random key
    # out: an action to take for each game

    if key == None:
      key = jax.random.PRNGKey(int(time.time()))

    scores = jax.random.uniform(key, shape=(*state[0].shape[:-1], self.config['width']))
    legal_cols = get_legal_cols(state)
    legal_scores = jnp.where(legal_cols, scores, jnp.nan)

    return jnp.nanargmax(legal_scores, axis=-1)[..., None]
