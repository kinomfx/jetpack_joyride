"""
train.py - Optimized CNN training script for JetPack JoyRide
Features: Frame stacking, 4-step training frequency, and efficient rendering.
"""

import argparse
import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dqn_agent import DQNAgent, preprocess_frame

def main():
    parser = argparse.ArgumentParser(description='Train CNN DQN agent on JetPack JoyRide')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--render-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--continue', type=str, dest='continue_from', default=None)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon-decay', type=float, default=0.9995)

    args = parser.parse_args()

    # Initialize Agent
    agent = DQNAgent(action_size=2, learning_rate=args.lr)
    agent.gamma = args.gamma
    agent.epsilon_decay = args.epsilon_decay

    if args.continue_from and os.path.exists(args.continue_from):
        agent.load(args.continue_from)

    try:
        from game_gym import GameGym
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        env = GameGym()
        scores, coins_collected, avg_scores = [], [], []

        # Frame stack for motion perception
        frame_stack = deque(maxlen=4)

        print("Starting Optimized CNN Training...")
        start_episode = agent.episode_count

        for episode in range(start_episode, start_episode + args.episodes):
            env.reset()

            # Initial frame capture
            first_frame = preprocess_frame(env.render_frame())
            for _ in range(4):
                frame_stack.append(first_frame)

            state = np.stack(list(frame_stack), axis=0)
            episode_reward, prev_coins, done = 0, 0, False
            render = (episode % args.render_every == 0)

            while not done:
                # 1. Action Selection
                action = agent.select_action(state)

                # 2. Environment Step
                _, reward, done, info = env.step(action)

                # 3. Process Next State
                next_frame_raw = env.render_frame()
                next_frame_proc = preprocess_frame(next_frame_raw)
                frame_stack.append(next_frame_proc)
                next_state = np.stack(list(frame_stack), axis=0)

                # 4. Reward Logic
                if done:
                    reward = -10.0
                else:
                    reward = 0.1
                    coin_diff = info['coins'] - prev_coins
                    if coin_diff > 0:
                        reward += 5.0

                episode_reward += reward
                prev_coins = info['coins']

                # 5. Store Experience
                agent.replay_buffer.push(state, action, reward, next_state, done)

                # 6. OPTIMIZATION: Train only every 4th step to remove lag
                if agent.total_steps % 4 == 0:
                    agent.train_step()

                state = next_state
                agent.total_steps += 1

                # 7. Render (Only if needed)
                if render:
                    env.render()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close(); agent.save("checkpoints/interrupted.pth"); return

            # Episode Stats
            agent.episode_count += 1
            scores.append(info['score'])
            coins_collected.append(info['coins'])

            if episode % agent.target_update == 0:
                agent.update_target_network()

            agent.decay_epsilon()
            avg_score = np.mean(scores[-100:]) if scores else 0
            avg_scores.append(avg_score)

            print(f"Ep {episode + 1} | Score: {info['score']} | ε: {agent.epsilon:.3f} | Avg: {avg_score:6.1f}")

            if (episode + 1) % args.save_every == 0:
                agent.save(f"checkpoints/cnn_ep_{episode + 1}.pth")

        agent.save("checkpoints/cnn_final.pth")
        env.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()