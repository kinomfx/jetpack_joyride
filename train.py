"""
train.py - Multi-Agent D3QN Optimized Training for JetPack JoyRide.
Implements co-evolution between a Protagonist and a Saboteur as per synopsis.pdf.
"""

import argparse
import os
import pygame
import numpy as np
import torch
from collections import deque
from d3qn_agent import D3QNAgent  # Ensure you update your agent class to D3QN
from dqn_agent import preprocess_frame

def main():
    parser = argparse.ArgumentParser(description='Competitive Co-Evolution D3QN Training')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--render-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--continue', type=str, dest='continue_from', default=None)
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"!!! SYSTEM CHECK: Training on {device} !!!")

    # 1. INITIALIZE DUAL AGENTS (Synopsis Requirement)
    # Protagonist: Actions [0: Nothing, 1: Jump]
    protagonist = D3QNAgent(action_size=2, learning_rate=args.lr, device=device)

    # Saboteur: Actions [0: Wait, 1: Missile High, 2: Missile Mid, 3: Missile Low]
    # Action size depends on your specific hazard implementation
    saboteur = D3QNAgent(action_size=4, learning_rate=args.lr, device=device)

    if args.continue_from and os.path.exists(args.continue_from):
        protagonist.load(args.continue_from + "_protagonist.pth")
        saboteur.load(args.continue_from + "_saboteur.pth")

    try:
        from game_gym import GameGym
        os.makedirs("checkpoints", exist_ok=True)
        env = GameGym()

        # Shared frame stack for pixel-based input
        frame_stack = deque(maxlen=4)

        print("Starting Competitive Adversarial Training...")

        for episode in range(protagonist.episode_count, protagonist.episode_count + args.episodes):
            render_this_episode = (episode % args.render_every == 0)
            env.reset(headless=not render_this_episode)

            # Build initial stack
            init_frame = preprocess_frame(env.render_frame())
            for _ in range(4): frame_stack.append(init_frame)
            state = np.stack(list(frame_stack), axis=0)

            done, prev_coins, ep_p_reward, ep_s_reward = False, 0, 0, 0

            while not done:
                # 2. SIMULTANEOUS ACTION SELECTION
                p_action = protagonist.select_action(state)
                s_action = saboteur.select_action(state)

                # 3. ENVIRONMENT STEP (Must handle two actions now)
                _, _, done, info = env.step(p_action, s_action)

                # 4. CAPTURE NEXT STATE
                next_frame = preprocess_frame(env.render_frame())
                frame_stack.append(next_frame)
                next_state = np.stack(list(frame_stack), axis=0)

                # 5. ZERO-SUM REWARD STRUCTURE[cite: 1]
                # Protagonist Reward
                p_reward = -10.0 if done else 0.1
                if info['coins'] > prev_coins: p_reward += 5.0

                # Saboteur Reward: Inverse of Protagonist[cite: 1]
                s_reward = 10.0 if done else -0.1

                prev_coins = info['coins']
                ep_p_reward += p_reward
                ep_s_reward += s_reward

                # 6. STORE EXPERIENCES
                protagonist.push(state, p_action, p_reward, next_state, done)
                saboteur.push(state, s_action, s_reward, next_state, done)

                # 7. CO-EVOLUTION TRAINING OPTIMIZATION[cite: 1]
                if protagonist.total_steps % 4 == 0:
                    protagonist.train_step()
                    saboteur.train_step()

                state = next_state
                protagonist.total_steps += 1

                if render_this_episode:
                    env.render(force_fps=True)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: return

            # Episode Updates
            protagonist.episode_count += 1
            if episode % 10 == 0:
                protagonist.update_target_network()
                saboteur.update_target_network()

            protagonist.decay_epsilon()
            saboteur.decay_epsilon()

            print(f"Ep {episode+1} | Score: {info['score']:4d} | P-Rew: {ep_p_reward:.1f} | S-Rew: {ep_s_reward:.1f}")

            if (episode + 1) % args.save_every == 0:
                protagonist.save(f"checkpoints/ep_{episode+1}_protagonist.pth")
                saboteur.save(f"checkpoints/ep_{episode+1}_saboteur.pth")

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()