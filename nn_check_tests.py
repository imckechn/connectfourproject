import haiku as hk
import pickle
import time
import agents
import jax, jax.numpy as jnp
import math

from simulators.Simulator import Simulator
from environment.connect_four import *
from config import default_config as config

if __name__ == '__main__':
    pl = get_piece_locations()

    def model(x):
        return hk.Sequential([
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(config['width'], w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))
        ])(x)

    model = hk.without_apply_rng(hk.transform(model))

    t_game = init_game(1)
    # if params exist, load them
    # if not then initialize randomly
    try:
        params = pickle.load(open("params.p", "rb"))
    except:
        key = jax.random.PRNGKey(int(time.time()))
        params = model.init(key, state_to_array_3(t_game, get_piece_locations()))

    rollout_policy = agents.GuidedRandomAgent(model, params)
    ucb_guide_agent = agents.UCBRolloutExpertAgent(15, 0, rollout_policy, 100)

    agent = agents.GuidedRolloutAgent(rollout_policy, batch_size=50)
    agent1 = agents.RandomAgent()
    agent2 = agents.RolloutAgent(batch_size=50)

    key = jax.random.PRNGKey(int(time.time()))
    sim = Simulator(init_game(100), [agent, agent2], key)
    results = sim.run(True)

    print(jnp.count_nonzero(results == -1))
    print(jnp.count_nonzero(results == 0))
    print(jnp.count_nonzero(results == 1))

    print(jnp.mean(results))
    