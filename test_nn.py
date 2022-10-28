import jax, jax.numpy as jnp
import time
import math

import haiku as hk
import optax as tx

from environment.connect_four import *
from config import default_config as config

def model(x):
    return hk.Sequential([
        hk.Linear(300), jax.nn.relu,
        hk.Linear(300), jax.nn.relu,
        hk.Linear(300), jax.nn.relu,
        hk.Linear(300), jax.nn.relu,
        hk.Linear(config['width'])
    ])(x)

model = hk.without_apply_rng(hk.transform(model))


def test_with_single_target_data(params, model, key, config):

    pl = get_piece_locations(config)
    loss = lambda params, x, y : tx.softmax_cross_entropy(model.apply(params, x), y).mean()
    value_and_grad_loss = jax.value_and_grad(loss)
    
    optimizer = tx.adam(1)
    opt_state = optimizer.init(params)

    # set the state to the initiali state and the target output should be to always choose the third column
    game_state = init_game(1)
    x = state_to_array(game_state, pl)
    dist = jnp.array([[1, 0, 0, 0, 0, 0, 0]])

    # do this lots to get very good 
    for i in range(100):
        game_state = init_game(100)
        x = state_to_array(game_state, pl)
        value, grads = value_and_grad_loss(params, x, dist)
        
        print(f'loss: {value}')
        

        updates, opt_state = optimizer.update(grads, opt_state)
        params = tx.apply_updates(params, updates)

    game_state = init_game(10000)
    x = state_to_array(game_state, pl)
    samples = jax.random.categorical(key, model.apply(params, x), shape=(10000,))
    print(samples)
    for i in range(7):
        print(i, " - ", jnp.count_nonzero(samples == i))
    


temp = init_game(1)
x = state_to_array(temp, get_piece_locations(config))
key = jax.random.PRNGKey(int(time.time()))
key, subkey1, subkey2 = jax.random.split(key, 3)

params = model.init(subkey1, x)
test_with_single_target_data(params, model, subkey2, config)