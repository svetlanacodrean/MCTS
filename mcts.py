# construct a binary tree of depth d = 12
from statistics import mode

import numpy as np
import random
from collections import defaultdict
import math
import matplotlib.pyplot as plt

distribution = []


class MCTS:
    def __init__(self, ucb_c=10):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.explored_nodes = dict()
        self.ucb_c = ucb_c

    def choose_successor(self, node):  # choose the best node successor
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.explored_nodes:
            return node.get_random_child()

        return self.uct_select(node)

    def do_rollout(self, node):  # train for one iteration
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backup(path, reward)

    def select(self, node):  # find an unexplored descendent of the node
        path = []
        while True:
            path.append(node)
            if node not in self.explored_nodes or node.is_terminal():
                return path
            unexplored_nodes = self.explored_nodes[node] - self.explored_nodes.keys()
            if unexplored_nodes:
                n = unexplored_nodes.pop()
                path.append(n)
                return path
            node = self.uct_select(node)

    def expand(self, node):  # add children in the explored nodes of the node
        if node in self.explored_nodes:  # already expanded
            return
        self.explored_nodes[node] = node.get_children()

    def simulate(self, node):  # returns node reward for a random simulation
        while not node.is_terminal():
            node = node.get_random_child()
        reward = node.reward()
        return reward

    def backup(self, path, reward):  # send the reward back
        for node in reversed(path):
            self.Q[node] += reward
            self.N[node] += 1

    def uct_select(self, node):  # when all children are already expanded
        def uct(n):
            return self.Q[n] / self.N[n] + self.ucb_c * math.sqrt(
                math.log(self.N[node]) / self.N[n])

        return max(self.explored_nodes[node], key=uct)


class Node:

    def __init__(self, info):

        self.left = None
        self.right = None
        self.info = info

    def insert_layer(self, is_last_layer):
        if self.left is None:
            self.insert_left_right(is_last_layer)
        else:
            self.left.insert_layer(is_last_layer)
            self.right.insert_layer(is_last_layer)

    def insert_left_right(self, ill):
        if ill == 0:
            self.left = Node(0)
            self.right = Node(0)
        else:
            x = np.random.uniform(0,100)
            y = np.random.uniform(0,100)
            self.left = Node(x)
            self.right = Node(y)
            distribution.append(x)
            distribution.append(y)

    def get_children(self):
        if self.is_terminal():
            return set()
        return {
            self.left,
            self.right
        }

    def get_random_child(self):
        if self.is_terminal():
            return None
        return random.choice([self.left, self.right])

    def is_terminal(self):
        if self.left is None:
            return True
        return False

    def reward(self):
        return self.info

    def print_tree(self):
        print(self.info, end =" ")
        if self.left:
            self.left.PrintTree()
        if self.right:
            self.right.PrintTree()


# create the tree
root = Node(0)
for i in range(11):
    root.insert_layer(0)
root.insert_layer(1)


# mcts search, repeat and collect the returns, find the max
mcts_results = []
for _ in range(50):
    tree = MCTS()
    current_root = root
    while not current_root.is_terminal():
        for _ in range(5):
            tree.do_rollout(current_root)
        current_root = tree.choose_successor(current_root)
    mcts_results.append(current_root.info)

print("Mcts results:           ", mcts_results)
print("Max mcts:                ", max(mcts_results))
print("Max of the distribution: ", max(distribution))

plt.plot(mcts_results, label="MCTS results")
mean = np.mean(mcts_results)
mode = mode(mcts_results)
print(mean)
print(mode)
x_coordinates = [0, 50]
y_coordinates = [mean, mean]
plt.plot(x_coordinates, y_coordinates, label="average MCTS")
y_coordinates = [mode, mode]
plt.plot(x_coordinates, y_coordinates, label="mode MCTS")
plt.legend()
plt.show()