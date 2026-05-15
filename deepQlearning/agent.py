import numpy as np
from nn import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, n_inputs, hidden_layers, n_actions, config):
        self.n_actions = n_actions

        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.tau = config.get("tau", 0.005)  # soft update : 0 = jamais, 1 = hard copy
        self.min_buffer = config.get("min_buffer", 1000)
        self.epsilon      = config.get("epsilon_start", 1.0)
        self.epsilon_min  = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        self.q_network = QNetwork(n_inputs, hidden_layers, n_actions)
        self.target_network = QNetwork(n_inputs, hidden_layers, n_actions)
        self.update_target()

        self.buffer = ReplayBuffer(config.get("buffer_size", 10000))

        self.step_count = 0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self.q_network.predict_action(state)

    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def train(self):
        if len(self.buffer) < self.min_buffer:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        S  = states.T
        S_ = next_states.T

        Q_next_q      = self.q_network.forward(S_)
        Q_next_target = self.target_network.forward(S_)

        Q_pred = self.q_network.forward(S)

        Q_target = Q_pred.copy()

        for i in range(self.batch_size):
            a = actions[i]
            if dones[i]:
                Q_target[a, i] = rewards[i]
            else:
                best_action = np.argmax(Q_next_q[:, i])
                Q_target[a, i] = rewards[i] + self.gamma * Q_next_target[best_action, i]

        loss = self.q_network.huber_loss(Q_pred, Q_target)
        self.q_network.backward(S, Q_target)
        self.q_network.update(self.lr)

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        self.soft_update()

        return loss

    def soft_update(self):
        for i in range(len(self.q_network.W)):
            self.target_network.W[i] = self.tau * self.q_network.W[i] + (1 - self.tau) * self.target_network.W[i]
            self.target_network.b[i] = self.tau * self.q_network.b[i] + (1 - self.tau) * self.target_network.b[i]

    def update_target(self):
        W, b = self.q_network.get_weights()
        self.target_network.set_weights(W, b)
