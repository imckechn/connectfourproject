from simulators.Simulator import Simulator
from config import default_config

import agents

from environment.connect_four import *

import jax, jax.numpy as jnp
import numpy as np

class TrainExpertSimulator(Simulator):
    def __init__(self, n_data, game_state, guided_agent, opponent, rollout_size=100, key=None, config=default_config):
        super().__init__(game_state=game_state, agents=[guided_agent, opponent], key=key, config=config)

        n_games = get_game_shape(self.game_state)

        self.data_counts = np.zeros((n_games, n_data, 7)) # times each action is chosen
        self.data_state = np.zeros((n_games, n_data, 4)) # position, mask, active, move#
        self.data_outcome = np.zeros((n_games, 1)) # final game outcome

        self.rollout_agent = agents.RolloutAgent(batch_size=rollout_size)
        self.rollout_agent.set_rollout_policy(self.agents[0])

        self.current_step = 0

    def step(self):
        self.key, subkey = jax.random.split(self.key)
        choices = self.agents[self.game_state[3]&1].choose(self.game_state, subkey)

        state, counts = self.game_state, self.rollout_agent.rollout_policy.counts
        self.data_counts[self.current_step] = counts
        self.data_state[self.current_step] = state

        self.game_state = play_move(self.game_state, choices.astype(jnp.uint64))

        self.current_step += 1

        

