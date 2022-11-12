from environment.connect_four import *
from config import default_config

class Agent():
  def __init__(self, config=default_config):
    self.config = config

  def choose(self, state, key=None, verbose=False):
    pass