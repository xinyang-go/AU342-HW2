import numpy as np
import random
from uilts import state2xyt


class Agent:
    ### START CODE HERE ###

    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 1
        self.random = 0.5
        self.record = []
        self.q_table = np.zeros([6, 6, 2, 4])  # u d l r

    def choose_action(self, state):
        x, y, t = state2xyt(state)
        max_val = -1e100
        max_action = []
        for i, q in enumerate(self.q_table[x, y, t]):
            if q > max_val:
                max_val = q
                max_action = []
            if q == max_val:
                max_action.append(i)
        action = np.random.choice(max_action, 1)[0]
        if random.random() < self.random or self.q_table[x, y, t].sum() == 0:
            action = np.random.choice(self.actions)
        return action

    def add_observation(self, s, a, r, s_, d):
        r -= 0.01
        self.record.append((s, a, r, s_))
        self.train_backward()
        if d:
            self.record = []
            self.random *= 0.92

    def train_backward(self):
        for i in range(len(self.record) - 1, -1, -1):
            s, a, r, s_ = self.record[i]
            x, y, t = state2xyt(s)
            x_, y_, t_ = state2xyt(s_)
            self.q_table[x, y, t, a] += 0.8 * (r + 0.9 * np.max(self.q_table[x_, y_, t_]) - self.q_table[x, y, t, a])

        ### END CODE HERE ###
