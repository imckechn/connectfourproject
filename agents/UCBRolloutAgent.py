from environment.connect_four import *
from config import default_config
from simulators.Simulator import Simulator

import time
import jax

import agents

class UCBRolloutAgent(agents.RolloutAgent):
    def __init__(self, time, temperature=1, batch_size=100, confidence_level=1, config=default_config):
        super().__init__(config=config, batch_size=batch_size)

        self.confidence_level = confidence_level
        self.time = time
        self.temperature = temperature

    def sample_all_arms(self, state, key):
        '''Performs batch rollout on each arm in all games'''
        n_state = expand_to_next_states(state, self.config)
        return (2 * get_player_turn(state) - 1) * jnp.sum(self.do_batch_rollout(n_state, key), axis=0)

    def sample_arm(self, state, arm, key):
        '''Performs batch rollout on specified arm in all games'''
        n_state = play_move(state, arm, self.config)
        return (2 * get_player_turn(state) - 1) * jnp.sum(self.do_batch_rollout(n_state, key), axis=0)

    def init_ucb(self, state):
        '''Initializes the counts, and score'''
        self.counts = jnp.zeros((*get_game_shape(state), self.config['width'], 1))
        self.total_scores = jnp.zeros_like(self.counts)

    def get_ucb_scores(self, state, time):
        '''Calculates the ucb score according to the UCB-selection rule'''
        return self.total_scores / self.counts + self.confidence_level * jnp.sqrt(jnp.log(time) / self.counts)

    def get_ucb_action(self, state, time):
        '''Gets the next legal action to choose based on the UCB-selection rule'''

        ucb_score = self.get_ucb_scores(state, time)
        legal = get_legal_cols(state, self.config)[..., None]
        legal_score = jnp.where(legal, ucb_score, jnp.nan)

        return jnp.nanargmax(legal_score, axis=-2).astype(jnp.uint64)
    
    def do_ucb_step(self, state, step, key):
        '''Does a full step of the UCB algorithm'''
        # ie. gets action, samples arm, then updates the action-value estimate and counts
        
        actions = self.get_ucb_action(state, step * self.batch_size)
        one_hot_action = jnp.squeeze(jax.nn.one_hot(actions, self.config['width']))
        score = self.sample_arm(state, actions, key)
        self.counts += (one_hot_action * self.batch_size)[..., None]
        self.total_scores += (one_hot_action * score)[..., None]

    def get_final_choice(self, shape, legal, key, verbose=False):
        
        logits = jnp.squeeze(1/self.temperature * jnp.log(self.counts))
        logits = jnp.where(legal, logits, jnp.finfo(float).min)

        choice = jax.random.categorical(key, logits, shape=shape)

        if verbose:
            print(jax.nn.softmax(logits))

        return choice.astype(jnp.uint64)
        
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

