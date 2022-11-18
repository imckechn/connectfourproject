import agents
from environment.connect_four import *
from simulators.Simulator import Simulator
import jax, jax.numpy as jnp
from compare_UCB_agents import load_UCB_agent_from_file
from itertools import permutations
import haiku as hk
import numpy as np
import time
import pickle
import pandas as pd

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

def get_agent_scores(agent_list, n_games, key):
    n_agents = len(agent_list)
    scores = np.zeros(shape=(n_agents, n_agents))
    for i, player1 in enumerate(agent_list):
        for j, player2 in enumerate(agent_list):
            print(f'Playing agents {i} v. {j}')
            key, subkey = jax.random.split(key)
            if player1 == player2:
                score = 0
            else:
                sim = Simulator(init_game(n_games), [player1, player2], subkey)
                results = sim.run()
                player0_wins = jnp.count_nonzero(results == -1)
                score = player0_wins / n_games

            scores[i, j] = score
            print(f'SCORE: {score}')
    
    return scores

def get_agent_scores_against(agent, agent_list, n_games, key):
    n_agents = len(agent_list)
    scores = np.zeros(shape=(n_agents, 2))
    for i, player2 in enumerate(agent_list):
        print(f'Playing agent v. {i}')
        key, subkey = jax.random.split(key)

        sim = Simulator(init_game(n_games), [agent, player2], subkey)
        results = sim.run()
        player0_wins = jnp.count_nonzero(results == -1)
        score = player0_wins / n_games

        scores[i, 0] = score
        print(f'SCORE: {score}')
    
    for i, player1 in enumerate(agent_list):
        print(f'Playing {i} v. agent')
        key, subkey = jax.random.split(key)

        sim = Simulator(init_game(n_games), [player1, agent], subkey)
        results = sim.run()
        player0_wins = jnp.count_nonzero(results == -1)
        score = player0_wins / n_games

        scores[i, 1] = score
        print(f'SCORE: {score}')
    
    return scores

def scores_to_ELO(sym_scores, loss_func):
    N = jnp.shape(sym_scores)[0]
    value_and_grad_loss = jax.jit(jax.value_and_grad(loss_func, argnums=0))
    ELO_ratings = jnp.zeros(N)
    learning_rate = 100
    N_epochs = 100_000

    for i in range(N_epochs):
        value, grad = value_and_grad_loss(ELO_ratings, sym_scores)
        ELO_ratings -= learning_rate * grad
        if i % (N_epochs // 3) == 0:
            print(f'Epoch: {i}:')
            print(f'Loss func: {value}')
        
    ELO_ratings = ELO_ratings - ELO_ratings[0]
    errors = 100*(expected_scores(ELO_ratings) - sym_scores)
    print(f'final percent errors: {errors}')
    print(f'row sums: {jnp.sum(errors, axis=0)}')
    print(f'final elos: {ELO_ratings}')

    return ELO_ratings

if __name__ == '__main__':
    jnp.set_printoptions(linewidth=100000, precision=4, edgeitems=5)

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
        load_UCB_agent_from_file('./datasets/ucb_net_v9/dataset_25_params.pk', model)
    ]

    n_games = 100
    key = jax.random.PRNGKey(int(time.time()))
    scores = get_agent_scores(agents, n_games, key)
    
    pickle.dump(scores.tolist(), open('./datasets/game_scores.pk', 'wb'))
    scores = np.array(pickle.load(open('./datasets/game_scores.pk', 'rb')))

    symmetric_scores = 0.5*(scores + (1 - scores).T)

    print(symmetric_scores)
    ELO = scores_to_ELO(symmetric_scores, L2_loss)

    #pickle.dump(ELO, open('./datasets/ucb_net_v9/elos2.pk', 'wb'))
    ELO = pickle.load(open('./datasets/ucb_net_v9/elos2.pk', 'rb'))

    rollout_winrate = symmetric_scores.T[0]
    
    fig, ax = plt.subplots()
   
    p = jnp.linspace(0.05, 0.8, 150)
    rating = 400*jnp.log10(p/(1-p))
    ax.set_title('ELO scores of UCB Expert Agent')
    ax.set_xlabel('Win rate vs Rollout Agent')
    ax.set_ylabel('ELO score')

    names = ['RO', 'G1', 'G4', 'G7', 'G10', 'G13', 'G16', 'G19', 'G25']
    y_off = [30, -40, 30, 40, -50, 40, 40, -40, 30]
    ax.plot(p, rating, color='gold')
    ax.scatter(rollout_winrate, ELO)

    for i, txt in enumerate(names):
        ax.annotate(txt, (rollout_winrate[i] - 0.015, ELO[i]+y_off[i]))
    
    ax.text(0.1, 100, 'G = Generation')

    fig = plt.gcf()
    fig.savefig('tournament_results_2.png', dpi=300)

    print(f'{jnp.squeeze(ELO)}')

