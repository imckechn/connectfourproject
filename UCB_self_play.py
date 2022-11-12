from environment.connect_four import *
from simulators.ExpertDataStore import ExpertDataStore

import jax, jax.numpy as jnp
import haiku as hk
import pandas as pd
import optax as tx

import time
import datetime

import pickle

from agents.UCBRolloutExpertAgent import UCBRolloutExpertAgent

def load_data_into_state(fname):
    # Load data from a csv into a game state and corresponding action counts
    # header: position, mask, active, move#, counts0, counts1, counts2, counts3, counts4, counts5, counts6

    start_time = time.perf_counter()
    data = pd.read_csv(fname)
    n_data = len(data)

    position = jnp.array(data.loc[:, 'position'].astype(jnp.uint64))
    mask = jnp.array(data.loc[:, 'mask'].astype(jnp.uint64))
    active = jnp.array(data.loc[:, 'active'].astype(jnp.uint64))
    move = jnp.array(data.loc[:, 'move#'].astype(jnp.uint64))
    counts = jnp.array(data.loc[:, 'counts0':'counts6'].astype(jnp.uint64))
    end_time = time.perf_counter()

    print(f'Loaded {n_data} samples in {end_time - start_time} seconds.')

    return (position[..., None], mask[..., None], active[..., None], move[..., None]), counts, n_data

def UCB_self_play(agent, num_games, cpu_folder, cpu_name, key=None):
    game_state = init_game(num_games)
    
    # stores the game information
    data_store = ExpertDataStore(42, get_all_shapes(game_state))
    
    if key == None:
        key = jax.random.PRNGKey(int(time.time()))

    # main game loop
    i = 0
    agent.temperature = 100

    start_time = time.perf_counter()
    while any_active_games(game_state):
        if i % 10 == 0:
            print(f'Move # {i}')
        
        key, subkey = jax.random.split(key)

        # agent choose action.
        column = agent.choose(game_state, subkey)

        # save the data to dataset
        data_store.store_data(game_state, agent.counts)

        # agent play action
        game_state = play_move(game_state, column)

        i += 1

        if i > 10:
            agent.temperature = 0.01
    end_time = time.perf_counter()

    print(f'Play finished in {end_time - start_time} seconds.')

    file_path = f'./datasets/{cpu_folder}/'
    file_name = f'dataset_{cpu_name}_{datetime.datetime.today().strftime("%Y-%m-%d")}'

    data_store.export_to_csv(file_path, file_name)

    return file_path, file_name

def UCB_self_play_train_model(fpath, fname, params, model, epochs):# helper array for converting state to array
    piece_locations = get_piece_locations()

    # load the data from the CSV
    game_states, counts, n_data = load_data_into_state(fpath + fname)

    # convert counts to probabilties
    temperature = 1
    logits = 1/temperature * jnp.log(counts)
    probabilities = jax.nn.softmax(logits)

    # initialize the loss function and it's gradient
    loss = lambda params, x, y : tx.softmax_cross_entropy(model.apply(params, x), y).mean()
    value_and_grad_loss = jax.value_and_grad(loss)

    # initialize the optimizer
    optimizer = tx.adam(0.001)
    opt_state = optimizer.init(params)

    # store the loss after each epoch for graphing
    data = {'epoch': [], 'CEL': []}
    
    print(f'Epochs: {epochs}')
    print(f'Sample Size: {n_data}')

    start_time = time.perf_counter()
    # do this lots to get very good 
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print(f'Epoch # {epoch}')

        x = state_to_array_3(game_states, piece_locations)
        
        value, grads = value_and_grad_loss(params, x, probabilities)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = tx.apply_updates(params, updates)
        
        data['epoch'].append(epoch)
        data['CEL'].append(float(value))

    end_time = time.perf_counter()

    print(f'Finished training, total elapsed time: {(end_time - start_time)/60.0:0.2f} minute(s).')

    return params, pd.DataFrame(data)

if __name__ == '__main__':

    key = jax.random.PRNGKey(int(time.time()))
    pl = get_piece_locations()

    def model(x):
        return hk.Sequential([
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(7, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))
        ])(x)


    model = hk.without_apply_rng(hk.transform(model))

    key,subkey = jax.random.split(key)
    t_game = init_game(1)

    # initialize the parameters of the network
    params = model.init(subkey, state_to_array_3(t_game, pl))

    for i in range(10):
        print(f'Generation: {i}')
        agent = UCBRolloutExpertAgent(1, model, params, 100)
        key,subkey = jax.random.split(key)
        fpath, fname = UCB_self_play(agent, 10, 'ucb_net', str(i), subkey)
        new_params, training_data = UCB_self_play_train_model(fpath, fname, params, model, 100)

        pickle.dump(new_params, open(fpath + fname + '_params.pk', 'wb'))
        pickle.dump(training_data, open(fpath + fname + '_training_data.pk', 'wb'))

        params = new_params