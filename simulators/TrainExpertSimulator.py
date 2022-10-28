from simulators.Simulator import Simulator
from simulators.ExpertDataStore import ExpertDataStore
from config import default_config

import agents

from environment.connect_four import *

import jax, jax.numpy as jnp
import numpy as np

import optax as tx

import pickle

class TrainExpertSimulator(Simulator):
    def __init__(self, epochs, sims_per_epoch, game_state, guided_agent, key=None, config=default_config):
        super().__init__(game_state=game_state, agents=[guided_agent, guided_agent], key=key, config=config)
        
        self.epochs = epochs
        self.sims_per_epoch = sims_per_epoch
        self.data = ExpertDataStore(sims_per_epoch*int(config['width'] * config['height']), get_all_shapes(game_state))
        self.piece_locations = get_piece_locations(self.config)

        # get the rollout agent model and parameters
        self.model = self.agents[0].rollout_agent.model
        self.params = self.agents[0].rollout_agent.params
        self.temperature = self.agents[0].rollout_agent.temperature

        # initialize loss, optimizer, and optimizer state
        self.loss = lambda params, nn_input, y : tx.softmax_cross_entropy(self.model.apply(params, nn_input), jnp.squeeze(y)).mean()
        self.optimizer = tx.adam(0.001)
        self.opt_state = self.optimizer.init(self.params)


    def step(self):
        self.key, subkey = jax.random.split(self.key)

        player_ix = get_player_turn(self.game_state)

        choices = self.agents[player_ix].choose(self.game_state, subkey)

        state, counts = self.game_state, self.agents[player_ix].counts
        self.data.store_data(state, counts)

        self.game_state = play_move(self.game_state, choices.astype(jnp.uint64))

    def update(self, state, counts):
        # update the parameters
        nn_input = state_to_array(state, self.piece_locations)
        logits = jnp.squeeze(1/self.temperature * jnp.log(counts))

        loss_value, grads = jax.value_and_grad(self.loss)(self.params, nn_input, jax.nn.softmax(logits))
        
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        return tx.apply_updates(self.params, updates), loss_value


    def train(self, verbose=False):

        for r in jnp.arange(self.epochs):
            print(f'Starting epoch {r} of {self.epochs}')
            for i in jnp.arange(self.sims_per_epoch):
                print(f'Starting sim {i} of {self.sims_per_epoch}')
                self.run(verbose)
                self.reset_simulator()

            for i in jnp.arange(self.data.data_pointer):
                state, counts = self.data.get_data(i)
                logits = jnp.squeeze(1/self.temperature * jnp.log(counts))

                self.params, loss_value = self.update(state, counts)
                pickle.dump(self.params, open("params.p", "wb"))

            print(f'cross-entropy Loss: {loss_value}')
            
            self.data.reset_pointer()

        return self.params


        

