import jax, jax.numpy as jnp
import time
import agents
import math

import haiku as hk
import optax as tx
import pandas as pd

import sys

from environment.connect_four import *
from config import default_config as config
from simulators.ExpertDataStore import ExpertDataStore

def generate_game_data(n_games=1, key=None, config=config, verbose=False):
    # generates data and UCB counts for specified number of games

    game_state = init_game(n_games)
    store = ExpertDataStore(config['width'] * config['height'], get_all_shapes(game_state), config)

    agent = [agents.UCBRolloutAgent(50, math.sqrt(2), 0.01, 100, config), agents.UCBRolloutAgent(50, math.sqrt(2), 0.01, 100, config)]
    
    if key == None:
        key = jax.random.PRNGKey(int(time.time()))

    for i in range(42):
        print(f'move # {i}')

        key, subkey = jax.random.split(key)

        player_ix = get_player_turn(game_state)
        
        choices = agent[player_ix].choose(game_state, subkey)
        new_game_state = play_move(game_state, choices.astype(jnp.uint64))
        counts = agent[player_ix].counts

        if verbose: print(f'{i} : counts [{player_ix}] : {jnp.squeeze(counts)} -> {jnp.squeeze(choices)}')

        store.store_data(game_state, counts)

        game_state = new_game_state
        
    if verbose: print(f'{get_winners(game_state, config)}')

    return store

if __name__ == "__main__":
    args = sys.argv
    
    assert len(args) > 1, 'must give file name'
    assert len(args) > 2, 'must include number of games'


    store = generate_game_data(n_games=int(args[2]))
    store.export_to_csv(args[1])
