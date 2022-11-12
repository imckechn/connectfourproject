import jax, jax.numpy as jnp
from environment.connect_four import *
from config import default_config as config
import haiku as hk
import pickle
import time
import agents
import math
import pandas as pd
from simulators.Simulator import Simulator
import seaborn as sns
import matplotlib.pyplot as plt

jnp.set_printoptions(linewidth=100000)

pl = get_piece_locations()

def model(x):
    return hk.Sequential([
        hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
        hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
        hk.Linear(config['width'], w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))
    ])(x)

model = hk.without_apply_rng(hk.transform(model))

t_game = init_game(100)
# if params exist, load them
# if not then initialize randomly
try:
    params = pickle.load(open("params.p", "rb"))
except:
    key = jax.random.PRNGKey(int(time.time()))
    params = model.init(key, state_to_array_3(t_game, pl))


legal = get_legal_cols(t_game)
logits = model.apply(params, state_to_array_3(t_game, pl))
legal_logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)


key = jax.random.PRNGKey(int(time.time()))

explorations = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]


data = {'exploration': [], 'score': [], 'SE': []}
for e in explorations:
    rollout_policy = agents.GuidedRandomAgent(model, params, exploration=e)
    grl_agent = agents.GuidedRolloutAgent(rollout_policy, batch_size=50)
    rl_agent = agents.RolloutAgent(batch_size=50)

    key, subkey = jax.random.split(key)
    sim = Simulator(init_game(1000), [grl_agent, rl_agent], key)
    results = sim.run()

    print(f'results: {e} : {jnp.mean(results)}')
    data['exploration'].append(e)
    data['score'].append(float(jnp.mean(results)))
    data['SE'].append(float(jnp.var(results)/math.sqrt(1000)))

plot = sns.lineplot(data=data, x='exploration', y='score')
plot.get_figure().savefig('plot.png')

data = pd.DataFrame(data)
plot = sns.lineplot(data=data, x='exploration', y='score')

plt.fill_between(data['exploration'], y1=data['score'] - data['SE']/2, y2=data['score'] + data['SE']/2)
plot.get_figure().savefig('exploration2.png')

pickle.dump(data, open("exploration_data2.pk", "wb"))