"""
train.py - Multi-Agent D3QN Optimized Training for JetPack JoyRide.
"""
import argparse
import os
import pygame
import numpy as np
import torch
from collections import deque
from d3qn_agent import D3QNAgent
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

    protagonist = D3QNAgent(action_size=2, learning_rate=args.lr, device=device)
    saboteur = D3QNAgent(action_size=5, learning_rate=args.lr, device=device)

    if args.continue_from and os.path.exists(args.continue_from):
        protagonist.load(args.continue_from + "_protagonist.pth")
        saboteur.load(args.continue_from + "_saboteur.pth")

    try:
        from game_gym import GameGym
        os.makedirs("checkpoints", exist_ok=True)
        env = GameGym()
        frame_stack = deque(maxlen=4)

        print("Starting Competitive Adversarial Training with Frame Skipping...")

        for episode in range(protagonist.episode_count, protagonist.episode_count + args.episodes):
            render_this_episode = (episode % args.render_every == 0)
            env.reset(headless=not render_this_episode)

            init_frame = preprocess_frame(env.render_frame())
            for _ in range(4): frame_stack.append(init_frame)
            state = np.stack(list(frame_stack), axis=0)

            done, prev_coins = False, 0
            ep_p_reward, ep_s_reward = 0, 0

            # Action holding variables
            p_action, s_action = 0, 0
            step_p_reward, step_s_reward = 0, 0

            while not done:
                # 1. CNN AWAKES ONLY EVERY 4 FRAMES
                if protagonist.total_steps % 4 == 0:
                    p_action = protagonist.select_action(state)
                    s_action = saboteur.select_action(state)
                    step_p_reward, step_s_reward = 0, 0 # Reset block rewards

                # 2. FAST PHYSICS STEP
                _, _, done, info = env.step(p_action, s_action)

                # 3. ACCUMULATE REWARDS
                p_rew = -10.0 if done else 0.1
                if info['coins'] > prev_coins: p_rew += 5.0
                s_rew = 10.0 if done else -0.1

                prev_coins = info['coins']
                step_p_reward += p_rew
                step_s_reward += s_rew
                ep_p_reward += p_rew
                ep_s_reward += s_rew

                # 4. PROCESS MEMORY AND TRAIN AT THE END OF THE SKIP (OR DEATH)
                if protagonist.total_steps % 4 == 3 or done:
                    next_frame = preprocess_frame(env.render_frame())
                    frame_stack.append(next_frame)
                    next_state = np.stack(list(frame_stack), axis=0)

                    protagonist.push(state, p_action, step_p_reward, next_state, done)
                    saboteur.push(state, s_action, step_s_reward, next_state, done)

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