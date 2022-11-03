import jax, jax.numpy as jnp
import time
import agents
import math

import haiku as hk
import optax as tx
import pandas as pd

from environment.connect_four import *
from config import default_config as config
from simulators.ExpertDataStore import ExpertDataStore

def model(x):
    return hk.Sequential([
        hk.Linear(300), jax.nn.relu,
        hk.Linear(300), jax.nn.relu,
        hk.Linear(config['width'])
    ])(x)

model = hk.without_apply_rng(hk.transform(model))


def test_with_single_target_data(params, model, epochs, outcome, learning_rate, n_samples, key, config):

    pl = get_piece_locations(config)
    loss = lambda params, x, y : tx.softmax_cross_entropy(model.apply(params, x), y).mean()
    value_and_grad_loss = jax.value_and_grad(loss)
    
    optimizer = tx.adam(learning_rate)
    opt_state = optimizer.init(params)

    # set the state to the initiali state and the target output should be to always choose the third column
    game_state = init_game(1)
    x = state_to_array(game_state, pl)

    # do this lots to get very good 
    for i in range(epochs):
        game_state = init_game(100)
        x = state_to_array(game_state, pl)
        value, grads = value_and_grad_loss(params, x, outcome)
        
        print(f'cross-entropy loss: {value}')
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = tx.apply_updates(params, updates)

    game_state = init_game(1)
    x = state_to_array(game_state, pl)
    samples = jax.random.categorical(key, model.apply(params, x), shape=(n_samples, 1))
    print()
    print("target", outcome)
    print("actual", jnp.array([jnp.count_nonzero(samples == i) for i in range(7)]) / n_samples)


temp = init_game(1)
x = state_to_array(temp, get_piece_locations(config))
key = jax.random.PRNGKey(int(time.time()))
key, subkey1, subkey2 = jax.random.split(key, 3)

params = model.init(subkey1, x)
test_with_single_target_data(params, model, 100, jnp.array([0.4, 0.1, 0, 0.1, 0.3, 0.1, 0]), 1, 10000, subkey2, config)

