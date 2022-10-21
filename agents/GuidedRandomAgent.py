from environment.connect_four import *
from config import default_config

import agents

import jax.numpy as jnp

class GuidedRandomAgent(agents.Agent):
    ''' Samples choices from a distributions that is given by a neural network'''

    def __init__(self, model, key, config=default_config):
        super().__init__(config=config)

        self.piece_locations = get_piece_locations(config)
        self.params = model.init(key, self.piece_locations.astype(float))
        self.model = model

    def get_model_params(self):
        return self.params

    def set_model_params(self, new_params):
        self.params = new_params

    def choose(self, state, key=None):
        '''chooses actions using the state'''
        # in: state - a 4-tuple with the current game state (position, mask, active, turn)
        #     key - the jax random key
        # out: an action to take for each game

        z = state_to_array(state, self.piece_locations)

        # logit is the unnormalized log probability (ln(p) - ln(1-p))
        # maps probabilities in (0,1) to (-infty, +infty)
        logits = self.model.apply(self.params, z)

        legal = get_legal_cols(state)

        # sets illegal actions ot the minimum possible float value so that it's never chosen
        legal_logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)

        print("distribution\n", jax.nn.softmax(legal_logits)[0])

        shape = get_game_shape(state)

        # sample from categorical distribution
        choice = jax.random.categorical(key, legal_logits, shape=shape)

        for i in range(7):
            print(i, " = ", jnp.count_nonzero(choice == i))

        return choice[..., None]
