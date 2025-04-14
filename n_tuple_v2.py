import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
import os
from student_agent import Game2048Env
import GPUtil
import gym_bandits
import gym
import torch
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


def rotate_90(positions):
    return [(y, 3 - x) for (x, y) in positions]

def reflect(positions):
    return [(x, 3 - y) for (x, y) in positions]


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)


    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        return [
            pattern,
            rotate_90(pattern),
            rotate_90(rotate_90(pattern)),
            rotate_90(rotate_90(rotate_90(pattern))),
            rotate_90(reflect(pattern)),
            rotate_90(rotate_90(reflect(pattern))),
            rotate_90(rotate_90(rotate_90(reflect(pattern))))
        ]


    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        # print("coords:", coords)
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total = 0
        for i, pattern in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                total += self.weights[i][feature]
        return total

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, pattern in enumerate(self.patterns):
            for sym_pattern in self.symmetry_patterns[i]:
                feature = self.get_feature(board, sym_pattern)
                self.weights[i][feature] += (alpha * delta / (8 * len(patterns)))


def save_checkpoint(approximator, episode, final_scores, success_flags, filename='checkpoint2.pkl'):
    checkpoint = {
        'approximator': approximator,
        'episode': episode,
        'final_scores': final_scores,
        'success_flags': success_flags
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at episode {episode + 1}")




def load_checkpoint(filename='checkpoint2.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Checkpoint loaded from episode {checkpoint['episode'] + 1}")
        return checkpoint['approximator'], checkpoint['episode'], checkpoint['final_scores'], checkpoint['success_flags']
    else:
        print("No checkpoint file found. Starting training from scratch.")
        return NTupleApproximator(board_size=4, patterns=patterns), 0, [], []
    