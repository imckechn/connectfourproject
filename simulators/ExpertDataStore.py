import numpy as np
import csv
from config import default_config
import os
from time import perf_counter

class ExpertDataStore():
    '''Handles the data storing and saving for the game state and action counts'''

    def __init__(self, n_data, shapes, config=default_config):
        self.config = config
        self.n_data = n_data

        self.initialize_data_store(shapes)

    def reset_pointer(self):
        self.data_pointer = 0

    def initialize_data_store(self, shapes):
        # initialize data stores for the game state, outcomes, and the move counts
        # this uses numpy, so that we can use assignment operator
        # numpy is also better here because it will store this data in CPU RAM
        #   -> it wont take up GPU RAM that is needed for the simulations.
 
        self.data_pointer = 0

        self.data_action_counts = np.zeros((self.n_data, *(shapes[0][:-1]), self.config['width'], 1), dtype=np.uint64) # times each action is chosen
        self.data_position = np.zeros((self.n_data, *(shapes[0])), dtype=np.uint64) # position of the current players pieces
        self.data_mask = np.zeros((self.n_data, *(shapes[1])), dtype=np.uint64) # position of all pieces
        self.data_active = np.zeros((self.n_data, *(shapes[2])), dtype=np.uint64) # list of active games
        self.data_move = np.zeros((self.n_data, shapes[3]), dtype=np.uint64) # move number
        
        # self.data_outcome = np.zeros((self.n_data, 1)) # final game outcome

    def get_data(self, pointer):
        pos = self.data_position[pointer]
        mask = self.data_mask[pointer]
        active = self.data_active[pointer]
        move = self.data_move[pointer]
        counts = self.data_action_counts[pointer]

        return (pos, mask, active, move), counts
    
    def store_data(self, state, counts):
        # store the game state to the current pointer location
        self.data_position[self.data_pointer] = state[0]
        self.data_mask[self.data_pointer] = state[1]
        self.data_active[self.data_pointer] = state[2]
        self.data_move[self.data_pointer] = state[3]
        self.data_action_counts[self.data_pointer] = counts
        self.data_pointer += 1

    def export_to_csv(self, fpath, fname):
        # TODO: PROFILE THIS FUNCTION

        if not os.path.exists(fpath):
            os.makedirs(fpath)

        start_time = perf_counter()
        with open(fpath + fname, 'w') as file:
            headers = ['position', 'mask', 'active', 'move#'] + [f'counts{i}' for i in range(self.config['width'])]
            writer = csv.writer(file)
            writer.writerow(headers)

            for i in range(self.data_pointer):
                pos = self.data_position[i].ravel()
                mask = self.data_mask[i].ravel()
                counts = self.data_action_counts[i].ravel().reshape((-1, 7))
                active = self.data_active[i].ravel()
                move = self.data_move[i]

                for a, p, m, c in zip(active, pos, mask, counts):
                    if a == 1:
                        writer.writerow([p, m, a, int(move)] + c.tolist())
        end_time = perf_counter()

        print(f'Dataset stored to {fpath + fname} in {end_time - start_time} seconds.')
