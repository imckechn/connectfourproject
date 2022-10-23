from simulators.Simulator import Simulator
from simulators.ExpertDataStore import ExpertDataStore
from config import default_config

import agents

from environment.connect_four import *

import jax, jax.numpy as jnp
import numpy as np

class TrainExpertSimulator(Simulator):
    def __init__(self, n_data, game_state, guided_agent, opponent, key=None, config=default_config):
        super().__init__(game_state=game_state, agents=[guided_agent, opponent], key=key, config=config)
        
        self.data = ExpertDataStore(n_data, get_all_shapes(game_state))

    def step(self):
        self.key, subkey = jax.random.split(self.key)

        player_ix = get_player_turn(self.game_state)

        choices = self.agents[player_ix].choose(self.game_state, subkey)

        if player_ix == 0:
            state, counts = self.game_state, self.agents[player_ix].counts
            self.data.store_state(state, counts)

        self.game_state = play_move(self.game_state, choices.astype(jnp.uint64))


        

