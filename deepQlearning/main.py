import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from agent import DQNAgent

config = {
    "gamma":         0.99,
    "lr":            0.01,
    "batch_size":    32,
    "buffer_size":   10000,
    "min_buffer":    1000,
    "epsilon_start": 1.0,
    "epsilon_min":   0.01,
    "epsilon_decay": 0.0005,
    "tau":           0.005,
}

env        = gym.make("CartPole-v1")
env_render = gym.make("CartPole-v1", render_mode="human")

n_inputs  = env.observation_space.shape[0]   # 4
n_actions = env.action_space.n               # 2

STATE_MEAN = np.array([0.0, 0.0, 0.0, 0.0])
STATE_STD  = np.array([2.4, 2.5, 0.21, 2.5])

def normalize(state):
    return (state - STATE_MEAN) / STATE_STD

agent = DQNAgent(n_inputs, hidden_layers=[16, 16], n_actions=n_actions, config=config)

n_episodes        = 1000
best_reward       = 0
total_reward_list = []
loss_history      = []
episode = 0
mean = 0

while mean != 500.0 :
    render = (episode + 1) % 50 == 0
    current_env = env #env_render if render else env

    state, _ = current_env.reset()
    state = normalize(state)
    total_reward = 0
    done = False
    episode_losses = []

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = current_env.step(action)
        done = terminated or truncated
        next_state = normalize(next_state)

        agent.store(state, action, reward, next_state, done)
        loss = agent.train()
        if loss is not None:
            episode_losses.append(loss)

        state = next_state
        total_reward += reward

    total_reward_list.append(total_reward)
    if episode_losses:
        loss_history.append(np.mean(episode_losses))

    if total_reward > best_reward:
        best_reward = total_reward

    if (episode + 1) % 10 == 0:
        mean = sum(total_reward_list[-50:]) / min(50, len(total_reward_list))
        print(f"episode {episode+1:4d} | reward: {total_reward:6.1f} | best: {best_reward:6.1f} | epsilon: {agent.epsilon:.3f} | mean_score {mean:6.1f}")
    episode += 1


env.close()
env_render.close()

# Graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(total_reward_list, alpha=0.4, color="steelblue", label="reward")
window = 20
moving_avg = np.convolve(total_reward_list, np.ones(window) / window, mode="valid")
ax1.plot(range(window - 1, len(total_reward_list)), moving_avg, color="navy", label=f"moyenne mobile ({window})")
ax1.set_xlabel("épisode")
ax1.set_ylabel("reward")
ax1.set_title("Reward par épisode")
ax1.legend()

ax2.plot(loss_history, alpha=0.4, color="tomato", label="MSE loss")
if len(loss_history) >= window:
    loss_avg = np.convolve(loss_history, np.ones(window) / window, mode="valid")
    ax2.plot(range(window - 1, len(loss_history)), loss_avg, color="darkred", label=f"moyenne mobile ({window})")
ax2.set_xlabel("épisode")
ax2.set_ylabel("MSE loss")
ax2.set_title("Loss par épisode")
ax2.legend()

plt.tight_layout()
plt.show()

# Sauvegarde des poids
np.save(os.path.join(os.path.dirname(__file__), "weights_W_32.npy"), np.array(agent.q_network.W, dtype=object))
np.save(os.path.join(os.path.dirname(__file__), "weights_b_32.npy"), np.array(agent.q_network.b, dtype=object))
print("Poids sauvegardés.")
