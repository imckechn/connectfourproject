from environment.connect_four import *
from config import default_config
from simulators.Simulator import Simulator

import time
import jax

import agents
import haiku as hk

class UCBRolloutExpertAgent(agents.UCBRolloutAgent):
    def __init__(self, time, confidence_level, key, temperature=2, batch_size=10, config=default_config):
        super().__init__(time=time, confidence_level=confidence_level, batch_size=batch_size, config=config)

        self.temperature = temperature
        self.rollout_agent = self.initialize_rollout_agent(key)

    def get_ucb_scores(self, time):
        '''Calculates the ucb score according to the UCB-selection rule'''
            

        return self.total_scores / self.counts + self.confidence_level * jnp.sqrt(jnp.log(time) / self.counts)

    def initialize_rollout_agent(self, key):
        def f(x):
            return hk.Sequential([
                hk.Linear(300), jax.nn.relu,
                hk.Linear(100), jax.nn.relu,
                hk.Linear(self.config['width'])
            ])(x)

        f = hk.without_apply_rng(hk.transform(f))

        return agents.GuidedRandomAgent(f, key, self.config)

    def choose(self, state, key=None):
        if key == None:
            key = jax.random.PRNGKey(int(time.time()))

        key, subkey = jax.random.split(key)
        self.init_ucb(state)

        self.total_scores += self.sample_all_arms(state, subkey)
        self.counts += self.batch_size

        for i in jnp.arange(self.config['width'], self.time):
            key, subkey = jax.random.split(key)
            self.do_ucb_step(state, i, subkey)

        return self.get_ucb_action(state, self.time)

