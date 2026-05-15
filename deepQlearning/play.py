import os
import pygame
import numpy as np
import gymnasium as gym
from nn import QNetwork
from visualisation import Visualisation

STATE_MEAN = np.array([0.0, 0.0, 0.0, 0.0])
STATE_STD  = np.array([2.4, 2.5, 0.21, 2.5])

def normalize(state):
    return (state - STATE_MEAN) / STATE_STD

net = QNetwork(n_inputs=4, hidden_layers=[64, 64], n_actions=2)
DIR = os.path.dirname(__file__)
net.W = list(np.load(os.path.join(DIR, "weights_W_32.npy"), allow_pickle=True))
net.b = list(np.load(os.path.join(DIR, "weights_b_32.npy"), allow_pickle=True))

env = gym.make("CartPole-v1", render_mode="rgb_array")

visu = Visualisation()
visu.init(4, [16, 16], 2)

for episode in range(5):
    state, _ = env.reset()
    state = normalize(state)
    total_reward = 0
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()

        Q = net.forward(state.reshape(-1, 1))
        action = int(np.argmax(Q))

        state, reward, terminated, truncated, _ = env.step(action)
        state = normalize(state)
        done = terminated or truncated
        total_reward += reward

        frame = env.render()
        visu.update(net.get_all_activations(), state)
        visu.update_output(Q)
        visu.draw(frame)
        visu.render()

    print(f"épisode {episode + 1} | reward: {total_reward}")

env.close()
pygame.quit()
