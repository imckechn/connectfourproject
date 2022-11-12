import jax, jax.numpy as jnp
import time
import agents
from environment.connect_four import *
import haiku as hk
import pickle
import math
from simulators.Simulator import Simulator
from config import default_config as config
jnp.set_printoptions(linewidth=100000)

pl = get_piece_locations()

def model(x):
    return hk.Sequential([
        hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
        hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
        hk.Linear(config['width'], w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))
    ])(x)

model = hk.without_apply_rng(hk.transform(model))

t_game = init_game(1)

key = jax.random.PRNGKey(int(time.time()))

# if params exist, load them
# if not then initialize randomly
try:
    params = pickle.load(open("params.p", "rb"))
except:
    params = model.init(key, state_to_array_3(t_game, pl))


key, subkey = jax.random.split(key)

expert = agents.UCBRolloutExpertAgent(100, model, params, batch_size=1)
rollout = agents.RolloutAgent()

sim = Simulator(init_game(10), [expert, rollout], subkey)
results = sim.run(verbose=True)

print(f'wins: {jnp.count_nonzero(results == -1)}')
print(f'ties: {jnp.count_nonzero(results == 0)}')
print(f'loss: {jnp.count_nonzero(results == 1)}')


