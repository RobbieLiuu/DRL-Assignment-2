# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import pickle
import os
import GPUtil
import gym_bandits
import torch
from my_td_mcts import TD_MCTS,TD_MCTS_Node
from game_2048env import Game2048Env
from game_2048env import NTupleApproximator

torch.cuda.empty_cache()


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

# Empty GPU cache before training
torch.cuda.empty_cache()

def get_gpu_with_most_memory():
    gpus = GPUtil.getGPUs()
    max_free_mem = -1
    gpu_id_with_max_mem = -1
    for gpu in gpus:
        free_mem = gpu.memoryFree
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            gpu_id_with_max_mem = gpu.id
    return gpu_id_with_max_mem, max_free_mem
approximator = None  

def load_approximator(filename='checkpoint2-20000.pkl'):
    """Load the trained N-Tuple Approximator from a checkpoint file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Approximator loaded from checkpoint (episode {checkpoint['episode'] + 1})")
        return checkpoint['approximator']
    else:
        print("No checkpoint file found. Please train the approximator first.")
        return None



def get_action(state, score):
    """
    Choose the best action based on the N-Tuple Approximator
    
    Args:
        state: Current board state (4x4 numpy array)
        score: Current game score
        
    Returns:
        action: 0 (up), 1 (down), 2 (left), or 3 (right)
    """
    global approximator
    
    # Load the approximator if not already loaded
    if approximator is None:
        approximator = load_approximator()
        if approximator is None:
            # Fallback to random strategy if approximator can't be loaded
            return random.choice([0, 1, 2, 3])
    
    # Create a temporary environment to simulate actions
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99)

    
    root = TD_MCTS_Node(state, env.score)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    print("TD-MCTS selected action:", best_act)

    return best_act



# def get_action(state, score):
#     """
#         action: 0 (up), 1 (down), 2 (left), or 3 (right)
#     """
#     global approximator
    
#     # Load the approximator if not already loaded
#     if approximator is None:
#         approximator = load_approximator()
#         if approximator is None:
#             return random.choice([0, 1, 2, 3])
#     # Initialize the game environment
#     env = Game2048Env()
#     env.board = state.copy()
#     env.score = score
    
#     legal_moves = [a for a in range(4) if env.is_move_legal(a)]
#     if not legal_moves:
#         return 0  # Return any action if no legal moves (game over)
    
#    # Choose the best action using N-Tuple approximator
#     best_value = float('-inf')
#     best_action = None
    
#     for action in legal_moves:
#         env_copy = copy.deepcopy(env)  # Create a copy to simulate move
#         next_state, _, _, _ = env_copy.step(action)
#         value = approximator.value(next_state)   # Evaluate state value
        
#         if value > best_value:
#             best_value = value
#             best_action = action
    
#     # If no best action was found (unlikely), pick a random legal move
#     if best_action is None:
#         best_action = random.choice(legal_moves)
#     # Apply the best action
#     return best_action
