"""
train.py - Main training script for DQN agent

Usage:
    python train.py                    # Start training
    python train.py --episodes 500     # Train for 500 episodes
    python train.py --eval model.pth   # Evaluate a trained model
    python train.py --continue model.pth  # Continue training from checkpoint
"""

import argparse
import os
from dqn_agent import train_dqn_agent, evaluate_agent, DQNAgent


def main():
    parser = argparse.ArgumentParser(description='Train or evaluate DQN agent on JetPack JoyRide')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train (default: 1000)')
    parser.add_argument('--render-every', type=int, default=50,
                        help='Render every N episodes (default: 50)')
    parser.add_argument('--save-every', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--eval', type=str, default=None,
                        help='Path to model for evaluation')
    parser.add_argument('--continue', type=str, dest='continue_from', default=None,
                        help='Path to checkpoint to continue training from')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate (default: 0.995)')

    args = parser.parse_args()

    # Evaluation mode
    if args.eval:
        if os.path.exists(args.eval):
            print(f"Evaluating model: {args.eval}")
            evaluate_agent(args.eval, num_episodes=10, render=True)
        else:
            print(f"Error: Model file not found: {args.eval}")
        return

    # Training mode
    print("=" * 60)
    print("DQN Training for JetPack JoyRide")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Epsilon Decay: {args.epsilon_decay}")
    print(f"Render Every: {args.render_every} episodes")
    print(f"Save Every: {args.save_every} episodes")

    if args.continue_from:
        print(f"Continuing from: {args.continue_from}")

    print("=" * 60)
    print()

    # Create agent
    agent = DQNAgent(state_size=27, action_size=2, learning_rate=args.lr)
    agent.gamma = args.gamma
    agent.epsilon_decay = args.epsilon_decay

    # Load checkpoint if continuing training
    if args.continue_from and os.path.exists(args.continue_from):
        agent.load(args.continue_from)
        print(f"Loaded checkpoint. Starting from episode {agent.episode_count}")
        print()

    # Start training
    try:
        from game_gym import GameGym
        from state_extractor import extract_state_vector
        import pygame
        import numpy as np
        import matplotlib.pyplot as plt

        # Create directories
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        # Initialize environment
        env = GameGym()

        # Training statistics
        scores = []
        coins_collected = []
        losses = []
        avg_scores = []

        print("Starting training...")
        print("-" * 60)

        start_episode = agent.episode_count

        for episode in range(start_episode, start_episode + args.episodes):
            # Reset environment
            state = env.reset()

            episode_reward = 0
            episode_loss = []
            prev_score = 0
            prev_coins = 0
            done = False
            render = (episode % args.render_every == 0)

            while not done:
                # Handle pygame events
                if render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nTraining interrupted by user")
                            env.close()
                            agent.save("checkpoints/dqn_interrupted.pth")
                            return

                # Select action
                action = agent.select_action(state)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Custom reward
                if done:
                    reward = -100.0
                else:
                    reward = 0.1
                    coin_diff = info['coins'] - prev_coins
                    if coin_diff > 0:
                        reward += 10.0 * coin_diff

                episode_reward += reward
                prev_score = info['score']
                prev_coins = info['coins']

                # Store transition
                agent.replay_buffer.push(state, action, reward, next_state, done)

                # Train
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                state = next_state
                agent.total_steps += 1

                # Render
                if render:
                    env.render()

            # Episode finished
            agent.episode_count += 1
            scores.append(info['score'])
            coins_collected.append(info['coins'])

            # Update target network
            if episode % agent.target_update == 0 and episode > 0:
                agent.update_target_network()

            # Decay epsilon
            agent.decay_epsilon()

            # Calculate statistics
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            avg_score = np.mean(scores[-100:]) if scores else 0
            avg_scores.append(avg_score)

            # Logging
            print(f"Ep {episode + 1:4d}/{start_episode + args.episodes} | "
                  f"Score: {info['score']:4d} | "
                  f"Coins: {info['coins']:2d} | "
                  f"Avg: {avg_score:6.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (episode + 1) % args.save_every == 0:
                agent.save(f"checkpoints/dqn_episode_{episode + 1}.pth")

                # Plot progress
                fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                axes[0].plot(scores, alpha=0.3, label='Episode Score')
                axes[0].plot(avg_scores, label='Avg Score (100 ep)', linewidth=2)
                axes[0].set_xlabel('Episode')
                axes[0].set_ylabel('Score')
                axes[0].set_title('Training Progress')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(coins_collected, alpha=0.6, color='gold')
                axes[1].set_xlabel('Episode')
                axes[1].set_ylabel('Coins')
                axes[1].set_title('Coins Collected')
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'plots/progress_ep{episode + 1}.png')
                plt.close()

        # Save final model
        agent.save("checkpoints/dqn_final.pth")
        env.close()

        print()
        print("=" * 60)
        print("Training completed!")
        print(f"Final average score: {avg_scores[-1]:.1f}")
        print(f"Best score: {max(scores)}")
        print(f"Model saved to: checkpoints/dqn_final.pth")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        agent.save("checkpoints/dqn_interrupted.pth")
        print("Progress saved to: checkpoints/dqn_interrupted.pth")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        agent.save("checkpoints/dqn_error.pth")


if __name__ == "__main__":
    main()