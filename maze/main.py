from maze_env import Maze
from RL_brain import Agent
import time

# from uilts import view

actions = ["up", "down", "right", "left"]

if __name__ == "__main__":
    ### START CODE HERE ###
    # This is an agent with random policy. You can learn how to interact with the environment through the code below.
    # Then you can delete it and write your own code.

    env = Maze()
    agent = Agent(actions=list(range(env.n_actions)))
    for episode in range(50):
        s = env.reset()
        episode_reward = 0
        while True:
            env.render()  # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
            # time.sleep(0.1)
            # view(agent.q_table)

            a = agent.choose_action(s)
            s_, r, done = env.step(a)

            agent.add_observation(s, a, r, s_, done)

            episode_reward += r
            s = s_

            if done:
                env.render()
                time.sleep(0.5)
                break
        print('episode:', episode, 'episode_reward:', episode_reward)

    ### END CODE HERE ###

    print('\ntraining over\n')
