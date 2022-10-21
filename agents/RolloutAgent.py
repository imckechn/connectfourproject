from environment.connect_four import *
from config import default_config
from simulators.Simulator import Simulator

import time
import jax

import agents

class RolloutAgent(agents.Agent):
    def __init__(self, config=default_config, batch_size=100):
        super().__init__(config=config)

        self.batch_size = batch_size
        self.rollout_policy = agents.RandomAgent()

    def set_rollout_policy(self, agent):
        self.rollout_policy = agent

    def do_batch_rollout(self, state, key):
        '''Runs the given states to the end of the games'''
        p_state = repeat_game(state, self.batch_size)
        sim = Simulator(p_state, agents=[self.rollout_policy, self.rollout_policy], key=key, config=self.config)
        return sim.run()
        
    def choose(self, state, key=None):
        '''chooses actions using the state'''
        # in: state - a 4-tuple with the current game state (position, mask, active, turn)
        #     key - the jax random key
        # out: an action to take for each game

        if key == None:
            key = jax.random.PRNGKey(int(time.time()))

        # expand our game state over all possible states
        n_states = expand_to_next_states(state)

        # do batch rollout over the next states
        results = self.do_batch_rollout(n_states, key)
        
        # calculate the scores
        mean_scores = jnp.squeeze(jnp.mean(results, axis=0))
        legal_scores = jnp.where(get_legal_cols(state), mean_scores, jnp.nan)

        player_ix = state[3]&1

        self.scores = legal_scores * (2 * player_ix - 1)

        return jnp.nanargmax(legal_scores * (2*player_ix - 1), axis=-1)[..., None]