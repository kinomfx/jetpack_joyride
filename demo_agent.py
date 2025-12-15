"""
demo_agent.py - Simple live demo of trained DQN agent (NO OVERLAY)

This script plays the game with your trained agent in real-time.
Overlay/statistics drawing has been removed for a clean demo view.

Usage:
    python demo_agent.py                           # Play with default model
    python demo_agent.py --model path/to/model.pth # Play with specific model
    python demo_agent.py --episodes 10             # Play 10 episodes
"""

import argparse
import os
import pygame
from dqn_agent import DQNAgent
from game_gym import GameGym


def play_demo(agent, env, episodes):
    """Play the game with the trained agent (no overlay)"""
    all_scores = []
    all_coins = []

    print("\n" + "=" * 50)
    print("DQN Agent Playing...")
    print("=" * 50)
    print("Press ESC to stop | Press SPACE to skip episode")
    print("=" * 50)
    print()

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        print(f"Episode {episode}/{episodes}...", end=" ", flush=True)

        while not done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return all_scores, all_coins
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return all_scores, all_coins
                    elif event.key == pygame.K_SPACE:
                        done = True  # Skip to next episode

            # Agent action
            action = agent.select_action(state, training=False)

            # Step
            next_state, reward, done, info = env.step(action)
            state = next_state

            # Render only the game (no overlay)
            env.render()
            pygame.display.update()

        # Record stats
        all_scores.append(info['score'])
        all_coins.append(info['coins'])

        print(f"Score: {info['score']:4d} | Coins: {info['coins']:2d}")

    return all_scores, all_coins


def main():
    parser = argparse.ArgumentParser(description='Demo trained DQN agent (no overlay)')
    parser.add_argument('--model', type=str, default='checkpoints/dqn_final.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Game speed multiplier (default: 1.0, use 0.5 for slow-mo)')

    args = parser.parse_args()

    # Check model
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("\nAvailable models:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith('.pth'):
                    print(f"   - checkpoints/{f}")
        return

    print("\n" + "=" * 25)
    print("     DQN Agent Live Demo")
    print("=" * 25)
    print(f"\n Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Speed: {args.speed}x")

    # Load agent
    print("\nLoading agent...", end=" ", flush=True)
    agent = DQNAgent(state_size=31, action_size=2)
    agent.load(args.model)
    agent.epsilon = 0
    print("Done!")

    # Setup environment
    env = GameGym()
    if args.speed != 1.0:
        env.set_fps(int(60 * args.speed))

    # Play
    try:
        scores, coins = play_demo(agent, env, args.episodes)

        # Summary
        if scores:
            print("\n" + "=" * 50)
            print("Session Summary")
            print("=" * 50)
            print(f"Episodes: {len(scores)}")
            print(f"Average Score: {sum(scores)/len(scores):.1f}")
            print(f"Best Score: {max(scores)}")
            print(f"Worst Score: {min(scores)}")
            print(f"Total Coins: {sum(coins)}")
            print("=" * 50)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted")
    finally:
        env.close()
        print("\nDemo ended\n")


if __name__ == "__main__":
    main()
