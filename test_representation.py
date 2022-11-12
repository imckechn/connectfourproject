import jax, jax.numpy as jnp
import time

from environment.connect_four import *

game = init_game(1)

pl = get_piece_locations()

print(state_to_array_3(game, pl))