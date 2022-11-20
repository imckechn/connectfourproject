import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import functools

jax.config.update("jax_enable_x64", True)

from config import default_config

# a game is represented by the following bitmap

# .  .  .  .  .  .  .
# 5 12 19 26 33 40 47
# 4 11 18 25 32 39 46
# 3 10 17 24 31 38 45
# 2  9 16 23 30 37 44
# 1  8 15 22 29 36 43
# 0  7 14 21 28 35 42
#
# The 'position' variable stores the position of the CURRENT PLAYERS pieces
# The 'mask' variable stores the position of all pieces on the board

# For a 6 X 7 game of connect four we use a 7 X 7 bitboard. The extra row on top is important.
# It is used to encode the bitboard into a single unique key. 
# The unique key for a given state is  position + mask + bottom_row_mask

# IMPORTANT: The position is from only the CURRENT PLAYERS perspective.
# To get the other players we do: position XOR mask

# Assumption: We assume that red plays on even moves, and black plays on odd moves
#               To check parity, we look at the right most bit on the move count (corresponding to 2^0 = 1)
#               eg. move&1 = 1 if move is odd, move&1 = 0 if move is even

# Remark: If the player makes an illegal move, we ignore the move and go to the next players turn
#         (It would be better to let the player redo the action) for now just ensure that our AI's
#         do not make illegal moves..

def get_piece_locations(config=default_config):
  return jnp.array([2 ** (col + config['width'] * row) for row in range(config['height']) for col in range(config['width'])], dtype=jnp.uint64)

@jax.jit
def state_to_array_42(state, piece_locations, config=default_config):
  '''Represents the game state in an (..., 42), array'''

  current_player = jnp.bitwise_not(is_empty_intersection(state[0], piece_locations))
  opponent = jnp.bitwise_not(is_empty_intersection(get_opponent_state(state)[0], piece_locations))
  return (current_player - 1 * opponent).astype(float)

@jax.jit
def state_to_array(state, piece_locations, config=default_config):
  '''Represents the game state in an (..., 84), array'''

  current_player = jnp.bitwise_not(is_empty_intersection(state[0], piece_locations))
  opponent = jnp.bitwise_not(is_empty_intersection(get_opponent_state(state)[0], piece_locations))
  return jnp.concatenate([current_player, opponent], axis=-1).astype(float)

@jax.jit
def state_to_array_126(state, piece_locations, config=default_config):
  '''Represents the game state in an (..., 126) array'''

  current_player = jnp.bitwise_not(is_empty_intersection(state[0], piece_locations))
  opponent = jnp.bitwise_not(is_empty_intersection(get_opponent_state(state)[0], piece_locations))
  mask = jnp.bitwise_not(is_empty_intersection(state[1], piece_locations))

  return jnp.concatenate([current_player, opponent, mask], axis=-1).astype(float)

@jax.jit
def is_empty_intersection(bb_a, bb_b):
  return jnp.bitwise_and(bb_a, bb_b) == 0

def init_game(games_shape=1, config=default_config):
  '''initializes the game state tuple to the start of the game'''
  # Rmk: games_shape can be a number with the amount of games you want
  #       OR you can specify a shape tuple. eg. 10000, (1000, 10), and (10, 10, 10, 10) are all valid

  position = jnp.expand_dims(jnp.zeros(games_shape, dtype = jnp.uint64), axis=-1)
  mask = jnp.expand_dims(jnp.zeros(games_shape, dtype = jnp.uint64), axis=-1)
  active = jnp.expand_dims(jnp.ones(games_shape, dtype = jnp.uint64), axis=-1)
  count = 0

  return (position, mask, active, count)

@jax.jit
def get_column_list(config=default_config):
  return jnp.linspace(0, config['width'] - 1, config['width'], dtype=jnp.uint64)

@jax.jit
def get_board_mask(config=default_config):
  '''get a bitmask with all bits on'''
  return (1 << (config['height'] + 1) * config['width']) - 1

@jax.jit
def top_mask(col, config=default_config):
  '''get a bitmask with a 1 at the top of given column'''
  return (1 << (config['height'] - 1)) << (col * (config['height'] + 1))

@jax.jit
def bottom_mask(col, config=default_config):
  '''get a bitmask with a 1 at the bottom of the given column'''
  return 1 << col * (config['height'] + 1)

@jax.jit
def bottom_row_mask(config=default_config):
  '''get a bitmask with all the bits at the bottom of the board on'''
  h1 = config['height'] + 1
  size1 = h1 * config['width']
  all1 = (1 << size1) - 1
  col1 = (1 << h1) - 1
  return jnp.array(all1 / col1, dtype=jnp.uint64)

@jax.jit
def column_mask(col, config=default_config):
  '''get a bitmask with the bits of the given column on'''
  return ((1 << config['height']) - 1) << (col * (config['height'] + 1))

@jax.jit
def can_play(state, col, config=default_config):
  '''check if we can play in the given column'''
  _, mask, _, _ = state
  #print("maskcanplay", mask.shape)
  #print("topmask", top_mask(col, config).shape)
  #print("masktop", ((mask & top_mask(col, config)) == 0).shape)
  return (mask & top_mask(col, config)) == 0

@jax.jit
def alignment(state, config=default_config):
  '''checks if there is an alignment for the current player'''
  position, mask, active, count = state

  m = position & (position >> (config['height'] + 1))
  horizontal = jnp.where(m & (m >> (2 * (config['height'] + 1))), 1, 0)

  m = position & (position >> config['height'])
  diag_a = jnp.where(m & (m >> (2 * config['height'])), 1, 0)

  m = position & (position >> (config['height'] + 2))
  diag_b = jnp.where(m & (m >> (2 * (config['height'] + 2))), 1, 0)

  m = position & position >> 1
  vertical = jnp.where(m & (m >> 2), 1, 0)
  
  alignment = horizontal | diag_a | diag_b | vertical
  
  return alignment

@jax.jit
def get_legal_cols(state, config=default_config):
  legal_cols = can_play(state, get_column_list(), config)
  return legal_cols

@jax.jit
def get_opponent_state(state):
  '''gets the bitboard from the opponents perspective'''
  position, mask, active, count = state

  opponent_state = (position ^ mask, mask, active, count)
  return opponent_state

def position_code(state, config=default_config):
  '''generates unique key for the game'''
  position, mask, active, count = state
  return jnp.squeeze(position + mask + bottom_row_mask(config)).item()

@jax.jit
def update_active_games(state, config=default_config):
  '''checks if the given game is still active and updates the active array'''
  position, mask, active, count = state

  opponent_state = get_opponent_state(state)
  active = 1 - (alignment(opponent_state, config) | alignment(state, config) | (count >= 41))

  return (position, mask, active, count)

@jax.jit
def any_active_games(state, config=default_config):
  position, mask, active, count = state
  return jnp.any(active)

@jax.jit
def play_move(state, col, config=default_config):
  '''plays a move in the given column'''
  # note: this switches the players perspective then adds the bitmask
  #       with only an active bit at the bottom of the chosen column to
  #       the 'mask' bitmask this shifts all the bits in that column left by 1.

  # col must be of shape (pre-shape, 1) or a single integer
  position, mask, active, count = state

  position = position ^ mask

  legal = can_play(state, col, config) & active

  new_mask = mask | (mask + bottom_mask(col.astype(jnp.uint64), config))

  mask = jnp.where(legal, new_mask, mask)

  state = (position, mask, active, count)
  _, _, active, _ = update_active_games(state, config)

  count = count + 1

  return (position, mask, active, count)

@jax.jit
def get_winners(state, config=default_config):
  '''gets the winners of the current board'''
  # -1 if red wins
  #  1 if black wins
  # if there are two winners we get 0 (covers invalid game)
  # if there are no winners we get 0 (covers tie/game not over)
  position, mask, active, count = state
  opponent_state = get_opponent_state(state)

  current_alignments = alignment(state)
  opponent_alignments = alignment(opponent_state) * (-1)

  whos_turn = count & 1
  win_value = 2 * whos_turn.astype(jnp.int32) - 1 # -1 if red win, 1 is black win

  # score will be -1 if red has alignment, 1 if black has alignment, 0 if tied, 0 if both have alignments (invalid game)
  total_score = current_alignments * win_value + opponent_alignments * win_value

  return total_score

@jax.jit
def get_game_shape(state):
  return state[0].shape[:-1]

def get_all_shapes(state):
  return (state[0].shape, state[1].shape, state[2].shape, (1))

@jax.jit
def get_player_turn(state):
  '''gets whos turn it currently is'''
  return state[3]&1

@jax.jit
def get_winning_columns(state, config=default_config):
  '''gets the first columns that would make the current player win immediately'''
  # -1 if there are no winning columns

  col_masks = column_mask(get_column_list())
  win_mask = get_winning_bitmask(state)
  winning_cols = jnp.where(col_masks & win_mask != 0, 1, 0)

  return winning_cols

@jax.jit
def get_threatening_columns(state, config=default_config):
  '''gets the first column that would make the current player win immediately'''
  opponent_state = get_opponent_state(state)
  return get_winning_columns(opponent_state)

@jax.jit
def get_winning_bitmask(state, config=default_config):
  '''gets a bitmask with 1s in positions that would immediately end the game'''
  pos, mask, _, _ = state

  h = config['height']

  # vertical
  r = (pos << 1) & (pos << 2) & (pos << 3)

  # horizontal
  p = (pos << (h + 1)) & (pos << 2 * (h+1))
  r = r | (p & (pos << 3 * (h + 1)))
  r = r | (p & (pos >> (h + 1)))
  p = p >> (3 * (h + 1))
  r = r | (p & (pos << (h + 1)))
  r = r | (p & (pos >> (3 * (h + 1))))

  # diag 1
  p = (pos << h) & (pos << 2 * h)
  r = r | (p & (pos << 3 * h))
  r = r | (p & (pos >> h))
  p = p >> (3 * h)
  r = r | (p & (pos << h))
  r = r | (p & (pos >> 3 * h))

  # diag 2
  p = (pos << (h + 2)) & (pos << 2 * (h + 2))
  r = r | (p & (pos << 3 * (h + 2)))
  r = r | (p & (pos >> (h + 2)))
  p = p >> (3 * (h + 2))
  r = r | (p & (pos << (h + 2)))
  r = r | (p & (pos >> 3 * (h + 2)))

  return r & (get_board_mask(config) ^ mask)

def expand_to_next_states(state, config=default_config):
  '''expands the bitboard to all possible next states'''
  
  position, mask, active, count = state

  p_position = jnp.repeat(jnp.expand_dims(position, -2), 7, axis = -2)
  p_mask = jnp.repeat(jnp.expand_dims(mask, -2), 7, axis = -2)
  p_active = jnp.repeat(jnp.expand_dims(active, -2), 7, axis = -2)

  p_state = (p_position, p_mask, p_active, count)

  action = jnp.linspace(0, config['width'] - 1, config['width'], dtype=jnp.uint64)
  p_state = play_move(p_state, action[:, None], config)

  return p_state

@functools.partial(jax.jit, static_argnums=1)
def repeat_game(state, repeats, config=default_config):
  '''adds axis and repeats the game_state along the axis repeat times'''
  # output shape: (repeats, pre-shape, 1)
  position, mask, active, count = state

  p_position = jnp.repeat(jnp.expand_dims(position, 0), repeats, axis=0)
  p_mask = jnp.repeat(jnp.expand_dims(mask, 0), repeats, axis=0)
  p_active = jnp.repeat(jnp.expand_dims(active, 0), repeats, axis=0)

  p_state = (p_position, p_mask, p_active, count)

  return p_state

def to_array(game_state, game_num, config=default_config):
    position, mask, active, count = game_state

    board_size = (config['height']+1) * config['width']
    pieces = jnp.reshape(2 << (jnp.array(jnp.linspace(0, board_size - 1, board_size), dtype=jnp.uint64) - 1), (config['height'] + 1, config['width'])).T
    pieces = pieces.at[0, 0].set(1)
    pieces = jnp.flip(pieces, 0)

    current_player = jnp.where(pieces & position[game_num] != 0, 1 + (count&1), 0)
    other_player = jnp.where(pieces & (position ^ mask)[game_num] != 0, 2 - (count&1), 0)
    
    return current_player + other_player

def draw_game(game_state, game_num = 0, player_color = ['red','black'], config=default_config):
  '''Use matplotlib to draw the board in text given the bitboards for player 1 and player 2'''
  
  position, mask, active, count = game_state

  fig, ax = plt.subplots() 
  plt.xlim(-0.5, config['width'] - 0.5)
  plt.ylim(config['height']+0.5, 0.5)
  ax.patch.set_facecolor('lightgrey')

  game_array = to_array(game_state, game_num)

  for y in range(1, config['height']+1):
    for x in range(config['width']):
      outline = plt.Circle((x,y), 0.45, color='white')
      ax.add_patch(outline)

      if game_array[y, x] > 0:
        piece = plt.Circle((x, y), 0.45, color=player_color[game_array[y, x] - 1])
        ax.add_patch(piece)
        
  plt.savefig('fig.png')