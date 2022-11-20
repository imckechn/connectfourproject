from environment.connect_four import *
from simulators.Simulator import Simulator

import jax, jax.numpy as jnp
import haiku as hk

import pickle
import time

from agents.RolloutAgent import RolloutAgent
from agents.GuidedRolloutAgent import GuidedRolloutAgent
from agents.GuidedRandomAgent import GuidedRandomAgent
from agents.RandomAgent import RandomAgent
from agents.UCBRolloutExpertAgent import UCBRolloutExpertAgent
from agents.PlayerAgent import PlayerAgent

if __name__ == '__main__':
    jnp.set_printoptions(linewidth=100000, precision=4, edgeitems=5)
    key = jax.random.PRNGKey(int(time.time()))
    pl = get_piece_locations()

    # define the model
    def model(x):
        return hk.Sequential([
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(7, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))
        ])(x)

    model = hk.without_apply_rng(hk.transform(model))

    key,subkey = jax.random.split(key)
    t_game = init_game(1)

    # load parameters from file
    params = pickle.load(open('datasets/ucb_net_v9/dataset_50_params.pk', 'rb')) # REALLY GOOD
    params2 = pickle.load(open('datasets/ucb_net_v9/dataset_25_params.pk', 'rb')) # GOOD
    #params = pickle.load(open('datasets/ucb_net_v9/dataset_1_params.pk', 'rb'))
    
    key, subkey = jax.random.split(key)

    guided_random = GuidedRandomAgent(model, params)
    guided_rollout = GuidedRolloutAgent(guided_random)
    rollout_agent = RolloutAgent(batch_size=1000)
    random_agent = RandomAgent()
    ucb_expert = UCBRolloutExpertAgent(100, model, params, 10)
    ucb_expert2 = UCBRolloutExpertAgent(100, model, params, 10)

    player_agent = PlayerAgent()

    sim = Simulator(init_game(1000), [rollout_agent, ucb_expert], subkey)

    while(any_active_games(sim.game_state)):
        print(jax.nn.softmax(model.apply(params, state_to_array_2(sim.game_state, pl))))
        sim.step()
        draw_game(sim.game_state)

    print(jnp.mean(get_winners(sim.game_state)))
