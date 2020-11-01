# -*- coding:utf-8 -*-
# DQN homework.
import os
import sys
import gym
import tqdm
import pylab
import random
import numpy as np
from env import create_env
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from gym import wrappers
from tensorboardX import SummaryWriter
from utils import *

# hyper-parameter.  
EPISODES = 5000


class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.9
        self.learning_rate = 0.01
        self.epsilon = 0.6
        self.epsilon_min = 0.1
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / EPISODES
        self.batch_size = 128
        self.train_start = 1000

        # create replay memory using deque
        self.maxlen = 10000
        self.memory = []
        self.head = 0

        # create main model
        self.model_target = self.build_model()
        self.model_eval = self.build_model()

    # approximate Q function using Neural Network
    # you can modify the network to get higher reward.
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model_eval.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        if len(self.memory) < self.maxlen:
            self.memory.append(None)
        self.memory[self.head] = (state, [action], [reward], next_state, [not done])
        self.head = (self.head + 1) % self.maxlen
        # epsilon decay.

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        state = np.concatenate([mini_batch[b][0] for b in range(batch_size)])
        action = np.concatenate([mini_batch[b][1] for b in range(batch_size)])
        reward = np.concatenate([mini_batch[b][2] for b in range(batch_size)])
        next_state = np.concatenate([mini_batch[b][3] for b in range(batch_size)])
        not_done = np.concatenate([mini_batch[b][4] for b in range(batch_size)])

        target = self.model_eval.predict(state)
        target_val = self.model_target.predict(next_state)

        target[range(batch_size), action] = reward + not_done * self.discount_factor * np.amax(target_val, axis=1)

        # and do the model fit!
        history = self.model_eval.fit(state, target, batch_size=self.batch_size, epochs=1, verbose=0)
        return history.history["loss"][0]

    def eval2target(self):
        self.model_target.set_weights(self.model_eval.get_weights())

    def save(self, filepath):
        self.model_eval.save(filepath)

    def load(self, filepath):
        self.model_eval = load_model(filepath)
        self.eval2target()


if __name__ == "__main__":
    # load the gym env
    env = create_env('MsPacman-ram-v0')
    # set  random seeds to get reproduceable result(recommended)
    set_random_seed(0)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    # create the agent
    agent = DQNAgent(state_size, action_size)
    if os.path.exists("dqn.h5"):
        agent.load("dqn.h5")
    # log the training result
    scores, episodes = [], []
    graph_episodes = []
    graph_score = []
    avg_length = 10
    sum_score = 0

    writer = SummaryWriter()

    # train DQN
    steps = 0
    for e in tqdm.trange(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.array(state).reshape([1, state_size])
        lives = 3
        while not done:
            dead = False
            while not dead:
                steps += 1
                # render the gym env
                if agent.render:
                    env.render()
                # get action for the current state
                action = agent.get_action(state)
                # take the action in the gym env, obtain the next state
                next_state, reward, done, info = env.step(action)
                next_state = np.array(next_state).reshape([1, state_size])
                # judge if the agent dead
                dead = info['ale.lives'] < lives
                lives = info['ale.lives']
                # update score value
                score += reward
                if dead: reward -= 50.
                reward /= 10.
                # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done or dead)
                # train the evaluation network
                if steps % 5 == 0:
                    loss = agent.train_model()
                    if loss is not None:
                        writer.add_scalar("loss", loss, steps)
                # go to the next state
                state = next_state

        # update the target network after some iterations.
        if e % 10 == 0:
            agent.eval2target()
            agent.save("dqn.h5")

        #
        agent.update_epsilon()

        # print info and draw the figure.
        if done:
            writer.add_scalar("score", score, e)
            writer.flush()
