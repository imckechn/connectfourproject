import haiku as hk
import optax as tx
import pandas as pd
import jax, jax.numpy as jnp
import pickle
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import time

from environment.connect_four import *
from config import default_config as config

jnp.set_printoptions(linewidth=100000, precision=4, edgeitems=5)

def load_data_into_state(fname):
    # Load data from a csv into a game state and corresponding action counts
    # header: position, mask, active, move#, counts0, counts1, counts2, counts3, counts4, counts5, counts6

    data = pd.read_csv(fname)
    n_data = len(data)

    position = jnp.array(data.loc[:, 'position'].astype(jnp.uint64))
    mask = jnp.array(data.loc[:, 'mask'].astype(jnp.uint64))
    active = jnp.array(data.loc[:, 'active'].astype(jnp.uint64))
    move = jnp.array(data.loc[:, 'move#'].astype(jnp.uint64))
    counts = jnp.array(data.loc[:, 'counts0':'counts6'].astype(jnp.uint64))

    print(f'Loaded {n_data} samples.')

    return (position[..., None], mask[..., None], active[..., None], move[..., None]), counts, n_data

def shuffle_states(game_states, counts, n_data, key, batch_axis=0):
    position, mask, active, move = game_states
    shuffle_indices = jax.random.permutation(key, jnp.arange(n_data))

    # shuffled the states
    shuffled_position = jnp.take_along_axis(position, shuffle_indices[..., None], batch_axis)
    shuffled_mask = jnp.take_along_axis(mask, shuffle_indices[..., None], batch_axis)
    shuffled_active = jnp.take_along_axis(active, shuffle_indices[..., None], batch_axis)
    shuffled_move = jnp.take_along_axis(move, shuffle_indices[..., None], batch_axis)
    shuffled_counts = jnp.take_along_axis(counts, shuffle_indices[..., None], batch_axis)

    return (shuffled_position, shuffled_mask, shuffled_active, shuffled_move), shuffled_counts

def train_model_from_csv(fname, epochs, model, params, key, batch_size=None, verbose=False):

    # helper array for converting state to array
    piece_locations = get_piece_locations(config)

    # load the data from the CSV
    game_states, counts, n_data = load_data_into_state(fname)

    # if no batch size is specified do full batch gradient descent
    if batch_size == None:
        batch_size = n_data

    # convert counts to probabilties
    probabilities = counts / counts.sum(axis=1)[..., None]

    # initialize the loss function and it's gradient
    loss = lambda params, x, y : tx.softmax_cross_entropy(model.apply(params, x), y).mean()
    value_and_grad_loss = jax.value_and_grad(loss)

    # initialize the optimizer
    optimizer = tx.adam(0.001)
    opt_state = optimizer.init(params)

    # store the loss after each epoch for graphing
    data = {'epoch': [], 'CEL': []}

    n_iter = n_data // batch_size
    
    print(f'Epochs: {epochs}')
    print(f'Sample Size: {n_data}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Iterations per Epoch: {n_iter}')

    start_time = time.perf_counter()
    # do this lots to get very good 
    for epoch in range(epochs):
        # shuffle the data
        key, subkey = jax.random.split(key)
        shuffled_game_states, shuffled_probabilites = shuffle_states(game_states, probabilities, n_data, subkey)

        print(f'Epoch # {epoch}')

        position, mask, active, move = shuffled_game_states
        for i in range(n_iter):
            if ((i / n_iter)*100) % 1 == 0:
                print(i / n_iter * 100, "%")
            batched_game_states = (position[i:i+batch_size], mask[i:i+batch_size], active[i:i+batch_size], move[i:i+batch_size])
            batched_probabilities = shuffled_probabilites[i:i+batch_size]

            x = state_to_array(batched_game_states, piece_locations)
            
            value, grads = value_and_grad_loss(params, x, batched_probabilities)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = tx.apply_updates(params, updates)

        x = state_to_array(game_states, piece_locations)
        
        data['epoch'].append(epoch)
        data['CEL'].append(float(loss(params, x, probabilities)))

    end_time = time.perf_counter()

    print(f'Finished training, total elapsed time: {(end_time - start_time)/60.0:0.2f} minute(s).')

    return params, pd.DataFrame(data)

if __name__ == "__main__":
    args = sys.argv
    
    assert len(args) > 1, 'must give file name'
    assert len(args) > 2, 'must give number of epochs'

    def model(x):
        return hk.Sequential([
            hk.Linear(100), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
            hk.Linear(config['width'])
        ])(x)


    model = hk.without_apply_rng(hk.transform(model))

    # if params exist, load them
    # if not then initialize randomly
    try:
        params = pickle.load(open("params.p", "rb"))
    except:
        key = jax.random.PRNGKey(int(time.time()))
        t_game = init_game(1)
        params = model.init(key, state_to_array(t_game, get_piece_locations()))
    
    key = jax.random.PRNGKey(int(time.time()))
    new_params, data = train_model_from_csv(args[1], int(args[2]), model, params, key, batch_size=None)

    # save new parameters to pickle
    pickle.dump(new_params, open("params.p", "wb"))

    # generate plot of the loss throughout training
    loss_plot = sns.lineplot(data = data, x='epoch', y='CEL')
    loss_plot.get_figure().suptitle('Cross-Entropy Loss throughout training')
    loss_plot.get_figure().savefig('loss_stochastic')
