import jax, jax.numpy as jnp

import time

from environment.connect_four import *

state = init_game((10,10, 1))
piece_locations = get_piece_locations()
print(state_to_array(state, piece_locations).shape)