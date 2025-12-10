"""
dqn_agent.py - Complete DQN implementation for JetPack JoyRide

This file contains the DQN agent, neural network, replay buffer, and training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt


class DQN(nn.Module):
    """Deep Q-Network for JetPack JoyRide"""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for training on JetPack JoyRide"""

    def __init__(self, state_size=27, action_size=2, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update = 10  # Update target network every N episodes

        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Training statistics
        self.episode_count = 0
        self.total_steps = 0

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.criterion(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        print(f"Model loaded from {filepath}")


def compute_reward(game, prev_score, prev_coins, collision):
    """
    Reward function for the game
    - Survival: small positive reward per frame
    - Coins: medium reward
    - Score increase: small reward
    - Collision: large negative reward
    """
    if collision:
        return -100.0

    reward = 0.1  # Survival bonus

    # Coin collection bonus
    coin_diff = game.coin_count - prev_coins
    if coin_diff > 0:
        reward += 10.0 * coin_diff

    # Score increase (much smaller than coin reward)
    score_diff = game.score - prev_score
    reward += 0.01 * score_diff

    return reward


def train_dqn_agent(num_episodes=1000, render_every=50, save_every=100):
    """
    Train the DQN agent on JetPack JoyRide

    Args:
        num_episodes: Number of episodes to train
        render_every: Render the game every N episodes
        save_every: Save checkpoint every N episodes
    """
    from game_gym import GameGym
    from state_extractor import extract_state_vector
    import pygame

    # Create directories for saving
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Initialize agent and environment
    agent = DQNAgent(state_size=27, action_size=2)
    env = GameGym()

    # Training statistics
    scores = []
    coins_collected = []
    losses = []
    avg_scores = []

    print("Starting training...")
    print(f"Device: {agent.device}")
    print(f"Episodes: {num_episodes}")
    print("-" * 50)

    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()

        episode_reward = 0
        episode_loss = []
        prev_score = 0
        prev_coins = 0
        done = False
        render = (episode % render_every == 0)

        while not done:
            # Handle pygame events (for rendering)
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return agent, scores

            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Custom reward calculation
            reward = compute_reward(env.game, prev_score, prev_coins, done)
            episode_reward += reward
            prev_score = env.game.score
            prev_coins = env.game.coin_count

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            agent.total_steps += 1

            # Render occasionally
            if render:
                env.render()

        # Episode finished
        agent.episode_count += 1
        scores.append(env.game.score)
        coins_collected.append(env.game.coin_count)

        # Update target network
        if episode % agent.target_update == 0 and episode > 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Calculate statistics
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        # Logging
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Score: {env.game.score:4d} | "
              f"Coins: {env.game.coin_count:2d} | "
              f"Avg Score: {avg_score:6.1f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (episode + 1) % save_every == 0:
            agent.save(f"checkpoints/dqn_episode_{episode + 1}.pth")
            plot_training_progress(scores, coins_collected, avg_scores, episode + 1)

    # Save final model
    agent.save("checkpoints/dqn_final.pth")
    plot_training_progress(scores, coins_collected, avg_scores, num_episodes)

    env.close()
    print("\nTraining completed!")
    return agent, scores


def plot_training_progress(scores, coins, avg_scores, episode):
    """Plot and save training progress"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot scores
    axes[0].plot(scores, alpha=0.3, label='Episode Score')
    axes[0].plot(avg_scores, label='Average Score (100 episodes)', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Training Progress: Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot coins
    axes[1].plot(coins, alpha=0.6, color='gold')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Coins Collected')
    axes[1].set_title('Training Progress: Coins')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'plots/training_progress_ep{episode}.png')
    plt.close()


def evaluate_agent(agent_path, num_episodes=10, render=True):
    """
    Evaluate a trained agent

    Args:
        agent_path: Path to saved model checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game
    """
    from game_gym import GameGym
    import pygame

    # Load agent
    agent = DQNAgent(state_size=27, action_size=2)
    agent.load(agent_path)
    agent.epsilon = 0  # No exploration during evaluation

    env = GameGym()
    scores = []
    coins = []

    print(f"\nEvaluating agent for {num_episodes} episodes...")
    print("-" * 50)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return scores, coins

            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)

            if render:
                env.render()

        scores.append(env.game.score)
        coins.append(env.game.coin_count)
        print(f"Episode {episode + 1}: Score = {env.game.score}, Coins = {env.game.coin_count}")

    env.close()

    print("-" * 50)
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Average Coins: {np.mean(coins):.1f}")
    print(f"Best Score: {max(scores)}")

    return scores, coins


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode
        if len(sys.argv) > 2:
            model_path = sys.argv[2]
        else:
            model_path = "checkpoints/dqn_final.pth"

        if os.path.exists(model_path):
            evaluate_agent(model_path, num_episodes=10, render=True)
        else:
            print(f"Model not found: {model_path}")
    else:
        # Training mode
        agent, scores = train_dqn_agent(
            num_episodes=1000,
            render_every=50,
            save_every=100
        )