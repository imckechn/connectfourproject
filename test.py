import jax, jax.numpy as jnp
import time

import math
from environment.connect_four import *

import agents
from simulators.Simulator import Simulator

state = init_game((10, 10))

print(get_all_shapes(state))