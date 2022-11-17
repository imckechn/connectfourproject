from environment.connect_four import *
from config import default_config
from simulators.Simulator import Simulator

import time
import jax

import agents

class UCBRolloutExpertAgent(agents.UCBRolloutAgent):
    '''UCB Rollout Expert Agent is UCB rollouts but the rollout_agent should be guided by a model'''

    def __init__(self, time, model, params, batch_size=100, config=default_config):
        super().__init__(time=time, confidence_level=1, batch_size=batch_size, config=config)

        self.piece_locations = get_piece_locations(self.config)
        self.model = model
        self.params = params
        self.rollout_agent = agents.GuidedRandomAgent(model, params)
        self.temperature = 0.1
        self.exploration_temp = 2.5

    def get_ucb_scores(self, state, time):
        '''Calculates the ucb score according to the UCB-selection rule'''
        # (..., 7, 1)

        # sum(counts, keepdims=True, axis=-2) -> (1,1,1)
        

        sum_actions = jnp.sum(self.counts, keepdims=True, axis=-2)
        #c_puctbase = 19652
        #c_puctinit = 2.5
        #c_puct = jnp.log((sum_actions + c_puctbase + 1) / c_puctbase + c_puctinit)
        #ucb_score = self.total_scores / self.counts + c_puct * self.nn_pred * jnp.sqrt(sum_actions) / (1 + self.counts)
        c_puct = 1
        ucb_score = self.total_scores / self.counts + c_puct * jnp.log(self.time * self.batch_size) * self.nn_pred / jnp.sqrt(1 + self.counts)
        return ucb_score

    def choose(self, state, key=None, verbose=False):
        if key == None:
            key = jax.random.PRNGKey(int(time.time()))

        key, subkey = jax.random.split(key)
        self.init_ucb(state)

        self.total_scores += self.sample_all_arms(state, subkey)
        self.counts += self.batch_size
        
        self.nn_pred = jax.nn.softmax(self.model.apply(self.params, state_to_array_2(state, self.piece_locations)) / self.exploration_temp)[..., jnp.newaxis]

        for i in jnp.arange(self.config['width'], self.time):
            key, subkey = jax.random.split(key)
            self.do_ucb_step(state, i, subkey)

        key, subkey = jax.random.split(key)
        shape = get_game_shape(state)
        legal = get_legal_cols(state)

        return self.get_final_choice(shape, legal, subkey, verbose=verbose)[..., None]

