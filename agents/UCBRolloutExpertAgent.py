from environment.connect_four import *
from config import default_config
from simulators.Simulator import Simulator

import time
import jax

import agents
import haiku as hk

class UCBRolloutExpertAgent(agents.UCBRolloutAgent):
    '''UCB Rollout Expert Agent is UCB rollouts but the rollout_agent should be guided by a model'''

    def __init__(self, time, confidence_level, rollout_agent, temperature=2, batch_size=10, config=default_config):
        super().__init__(time=time, confidence_level=confidence_level, batch_size=batch_size, config=config)

        self.temperature = temperature
        self.rollout_agent = rollout_agent

    def get_ucb_scores(self, state, time):
        '''Calculates the ucb score according to the UCB-selection rule'''
        return self.total_scores / self.counts + self.confidence_level * (self.rollout_agent.get_model_predictions(state)[..., jnp.newaxis] / (1 + self.counts))

    def choose(self, state, key=None, verbose=False):
        if key == None:
            key = jax.random.PRNGKey(int(time.time()))

        key, subkey = jax.random.split(key)
        self.init_ucb(state)

        self.total_scores += self.sample_all_arms(state, subkey)
        self.counts += self.batch_size

        for i in jnp.arange(self.config['width'], self.time):
            key, subkey = jax.random.split(key)
            self.do_ucb_step(state, i, subkey)

        key, subkey = jax.random.split(key)
        shape = get_game_shape(state)
        legal = get_legal_cols(state)

        return self.get_final_choice(shape, legal, subkey, verbose=verbose)[..., None]

