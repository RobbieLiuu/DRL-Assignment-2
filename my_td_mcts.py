import copy
import random
import math
import numpy as np
from game_2048env import Game2048Env
# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_val = -float("inf")
        best_child = None
        parent = node
        for child in node.children.values():
            if child.visits == 0:
                UCT_val = float("inf")
            else:
                Q = child.total_reward / child.visits
                UCT_val = Q + self.c * math.sqrt(math.log(parent.visits) / child.visits)
            if UCT_val > best_val:
                best_child = child
                best_val = UCT_val
        return best_child


    def evaluate_afterstate(self, env):
        legal_actions = [a for a in range(4) if env.is_move_legal(a)]
        if not legal_actions:
            return 0

        max_value = float('-inf')
        for action in legal_actions:
            env_copy = copy.deepcopy(env)
            afterstate, _, _, _ = env_copy.step(action, add_random=False)
            if self.approximator.value(afterstate) is not None:
              value = self.approximator.value(afterstate)
            if value >= max_value:
                max_value = value

        return max_value


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        total_reward = 0
        cur_deteriorate_rate = 1
        cur_depth = 0
        while cur_depth < depth:
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            selected_action = random.choice(legal_moves)
            previous_score = sim_env.score
            _, _, is_terminal, _ = sim_env.step(selected_action)
            total_reward += cur_deteriorate_rate * (sim_env.score - previous_score)
            cur_deteriorate_rate *= self.gamma
            cur_depth += 1 
            if is_terminal:
                return total_reward

        estimated_value = self.evaluate_afterstate(sim_env)
        total_reward = total_reward + cur_deteriorate_rate * estimated_value

        return total_reward





    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent


    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.

        while node.fully_expanded() == True and node.children:
            node = self.select_child(node)
            sim_env.step(node.action)

        # TODO: Expansion: If the node is not terminal, expand an untried action.

        if node.fully_expanded() == False:
            selected_action = random.choice(node.untried_actions)
            sim_env.step(selected_action)
            newNode = TD_MCTS_Node(sim_env.board.copy(), sim_env.score, parent=node, action=selected_action)
            node.children[selected_action] = newNode
            node.untried_actions.remove(selected_action)
            node = newNode


        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
