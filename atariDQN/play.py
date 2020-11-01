import os
import time
import numpy as np
from atariDQN import DQNAgent
from env import create_env

if __name__ == "__main__":
    env = create_env('MsPacman-ram-v0')
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    # create the agent
    agent = DQNAgent(state_size, action_size)
    assert os.path.exists("dqn.h5")
    agent.load("dqn.h5")
    agent.epsilon = 0

    done = False
    state = env.reset()
    state = np.reshape(state, [1, state_size]) / 255.
    while not done:
        time.sleep(0.01)
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size]) / 255.
        state = next_state
