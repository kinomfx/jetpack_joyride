import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DuelingVisualQNetwork(nn.Module):
    def __init__(self, action_size):
        super(DuelingVisualQNetwork, self).__init__()
        # CNN layers to process the 4-frame stack (84x84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Dueling Streams: Value and Advantage
        # Value stream: Estimates the base value of being in a state V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream: Estimates the benefit of each specific action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine streams using the standard Dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class D3QNAgent:
    def __init__(self, action_size, learning_rate=1e-4, gamma=0.99, device="cpu"):
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.episode_count = 0
        self.total_steps = 0
        self.target_update = 10  # Synchronize with train.py episodes[cite: 1]

        # Double DQN setup: Main network for learning, Target for stable estimation[cite: 1]
        self.q_network = DuelingVisualQNetwork(action_size).to(device)
        self.target_network = DuelingVisualQNetwork(action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=20000)

    def push(self, state, action, reward, next_state, done):
        """Fixed: Adds transition to the deque using .append()[cite: 1]"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """Epsilon-greedy action selection for exploration[cite: 1]"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Double DQN Logic[cite: 1]:
        # 1. Use Local Network to find the current Q-values
        current_q = self.q_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            # 2. Use Local Network to select the BEST action for the NEXT state
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # 3. Use Target Network to evaluate that action (reduces overestimation bias)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copies weights to the target network for stability[cite: 1]"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Gradually reduces randomness in actions[cite: 1]"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())