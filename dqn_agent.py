"""
dqn_agent.py - CNN-based DQN implementation for JetPack JoyRide
Updated for Step 1: Image-based processing with frame stacking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import cv2
import os

class DQN_CNN(nn.Module):
    """Convolutional Neural Network for processing game screens"""

    def __init__(self, action_size):
        super(DQN_CNN, self).__init__()
        # Input shape: (Batch, 4, 84, 84) - 4 stacked grayscale frames
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # After 84x84 input passes through convs, the feature map is 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """Experience replay buffer for image-based DQN"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state is expected to be a (4, 84, 84) numpy array
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """CNN-based DQN Agent for training on JetPack JoyRide"""

    def __init__(self, action_size=2, learning_rate=0.00025):
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # Slower decay for more complex CNN learning
        self.batch_size = 32
        self.target_update = 10  # episodes

        # Networks
        self.policy_net = DQN_CNN(action_size).to(self.device)
        self.target_net = DQN_CNN(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer (Image buffers take more RAM, keep capacity in mind)
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.episode_count = 0
        self.total_steps = 0

    def select_action(self, state, training=True):
        """Select action using image stack as input"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            # state is (4, 84, 84), add batch dimension -> (1, 4, 84, 84)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform training on a batch of image stacks"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping helps stability in CNNs
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }, filepath)
        print(f"CNN Model saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        print(f"CNN Model loaded from {filepath}")

def preprocess_frame(frame):
    """Convert RGB frame to 84x84 Grayscale for the CNN"""
    if frame is None:
        return np.zeros((84, 84), dtype=np.float32)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 2. Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    # 3. Normalize to [0, 1]
    return resized.astype(np.float32) / 255.0