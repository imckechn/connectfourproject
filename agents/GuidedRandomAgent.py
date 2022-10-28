from environment.connect_four import *
from config import default_config

import agents

import jax.numpy as jnp

class GuidedRandomAgent(agents.Agent):
    ''' Samples choices from a distributions that is given by a neural network'''

    def __init__(self, model, params, temperature=0.00001, config=default_config):
        super().__init__(config=config)

        self.piece_locations = get_piece_locations(config)
        self.params = params
        self.model = model
        self.temperature = temperature

    def get_model_predictions(self, state):
        '''Passes the given state through the neural network'''
        z = state_to_array(state, self.piece_locations)

        return self.model.apply(self.params, z)

    def choose(self, state, key=None):
        '''chooses actions using the state'''
        # in: state - a 4-tuple with the current game state (position, mask, active, turn)
        #     key - the jax random key
        # out: an action to take for each game

        # logit is the unnormalized log probability (ln(p) - ln(1-p))
        # maps probabilities in (0,1) to (-infty, +infty)
        logits = self.get_model_predictions(state)

        legal = get_legal_cols(state)

        # sets illegal actions ot the minimum possible float value so that it's never chosen
        legal_logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)

        shape = get_game_shape(state)

        # sample from categorical distribution
        choice = jax.random.categorical(key, legal_logits, shape=shape)

        return choice[..., None]
