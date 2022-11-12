from environment.connect_four import *
from config import default_config
import agents

import time
import jax

class RandomPlusAgent(agents.Agent):
  def __init__(self, config=default_config):
    super().__init__(config=config)

  def choose(self, state, key=None, verbose=False):
    '''chooses actions using the state'''
    # in: state - a 4-tuple with the current game state (position, mask, active, turn)
    #     key - the jax random key
    # out: an action to take for each game
    if key == None:
      key = jax.random.PRNGKey(int(time.time()))

    random_scores = jax.random.uniform(key, shape=(*state[0].shape[:-1], self.config['width']))
    threats = get_threatening_columns(state, self.config)
    wins = get_winning_columns(state, self.config)

    adjusted_scores = 1000*wins + 100*threats + random_scores
    legal_scores = jnp.where(get_legal_cols(state), adjusted_scores, jnp.nan)

    return jnp.nanargmax(legal_scores, axis=-1)[..., None]