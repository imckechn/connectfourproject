import agents
from environment import *
from simulators.Simulator import Simulator
import jax, jax.numpy as jnp
from compare_UCB_agents import load_UCB_agent_from_file
from itertools import permutations
import haiku as hk

# ELO SCORES
# ELO formula for expected score for player A when the ratings are R_a, and R_b is

# E_a = 1/(1 + 10^{(R_b - R_a)/400})

@jax.jit
def expected_scores(ELO_ratings):
    #input: elo (N,)
    #output: (n,n) matrix with expected tournament scores according to ELO formula

    return 1 / (1 + jnp.power(10, (ELO_ratings[jnp.newaxis, :] - ELO_ratings[:, jnp.newaxis]) / 400))

@jax.jit
def L2_loss(ELO_ratings, actual_scores):
    return jnp.mean((expected_scores(ELO_ratings) - actual_scores) ** 2)

if __name__ == '__main__':

    # define the model
    def model(x):
        return hk.Sequential([
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(100, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')), jax.nn.relu,
            hk.Linear(7, w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'))
        ])(x)

    model = hk.without_apply_rng(hk.transform(model))
    agents = [
        agents.RolloutAgent(batch_size=200),
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_1_params.pk', model),
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_4_params.pk', model),
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_7_params.pk', model),
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_10_params.pk', model),
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_13_params.pk', model),
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_16_params.pk', model),
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_19_params.pk', model),
    ]

    p = permutations(agents)

    for j in list(p):
        print(j)

    ELO_ratings = jnp.array([100, 500, 1])
    print(expected_scores(ELO_ratings))