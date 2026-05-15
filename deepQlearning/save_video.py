import os
import cv2
import pygame
import numpy as np
import gymnasium as gym
from nn import QNetwork
from visualisation import Visualisation
from tqdm import tqdm

STATE_MEAN = np.array([0.0, 0.0, 0.0, 0.0])
STATE_STD  = np.array([2.4, 2.5, 0.21, 2.5])

def normalize(state):
    return (state - STATE_MEAN) / STATE_STD

DIR = os.path.dirname(__file__)

net = QNetwork(n_inputs=4, hidden_layers=[16, 16], n_actions=2)
net.W = list(np.load(os.path.join(DIR, "weights_W_32.npy"), allow_pickle=True))
net.b = list(np.load(os.path.join(DIR, "weights_b_32.npy"), allow_pickle=True))

env = gym.make("CartPole-v1", render_mode="rgb_array")

visu = Visualisation()
visu.init(4, [16, 16], 2)
clock = pygame.time.Clock()

# Setup video
FPS        = 60
output     = os.path.join(DIR, "replay.mp4")
fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
writer     = cv2.VideoWriter(output, fourcc, FPS, (visu.screen_w, visu.screen_h))

state, _ = env.reset()
state = normalize(state)
done = False
total_reward = 0

for i in tqdm(range(100)):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

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
    pygame.display.flip()

    pixels = pygame.surfarray.array3d(visu.screen)
    pixels = pixels.swapaxes(0, 1)
    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    writer.write(pixels)
    clock.tick(2)

writer.release()
env.close()
pygame.quit()

print(f"reward: {total_reward}")
print(f"vidéo sauvegardée : {output}")
