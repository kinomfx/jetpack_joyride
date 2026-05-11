"""
eval.py - Pure Intelligence Evaluation (Zero Randomness)
Tests the Nash Equilibrium stability of the trained agents.
"""

import argparse
import os
import pygame
import numpy as np
import torch
from collections import deque

from d3qn_agent import D3QNAgent
from dqn_agent import preprocess_frame
from game_gym import GameGym


def main():
    parser = argparse.ArgumentParser(description='Evaluate Trained Co-Evolution Agents')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation games to watch')
    parser.add_argument('--model-ep', type=int, required=True,
                        help='Which checkpoint episode to load (e.g., 600, 1000)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"!!! SYSTEM CHECK: Evaluating on {device} !!!")

    # Initialize Agents (Sizes must match train.py)
    protagonist = D3QNAgent(action_size=2, device=device)
    saboteur = D3QNAgent(action_size=5, device=device)

    # File paths for the specified checkpoint
    p_path = f"checkpoints/ep_{args.model_ep}_protagonist.pth"
    s_path = f"checkpoints/ep_{args.model_ep}_saboteur.pth"

    if not os.path.exists(p_path) or not os.path.exists(s_path):
        print(f"Error: Could not find checkpoint files for episode {args.model_ep}")
        print(f"Looked for: \n- {p_path}\n- {s_path}")
        return

    # Load weights
    protagonist.load(p_path)
    saboteur.load(s_path)

    # SET NETWORKS TO EVALUATION MODE
    protagonist.q_network.eval()
    saboteur.q_network.eval()

    # LOCK EPSILON TO 0.0 (Pure Intelligence, No Randomness)
    protagonist.epsilon = 0.0
    saboteur.epsilon = 0.0
    protagonist.epsilon_decay = 1.0  # Prevent any decay logic from changing it
    saboteur.epsilon_decay = 1.0

    env = GameGym()
    frame_stack = deque(maxlen=4)

    print(f"\nStarting Evaluation for Checkpoint: {args.model_ep}")
    print("--------------------------------------------------")

    for episode in range(args.episodes):
        # Force rendering so you can watch the agents perform
        env.reset(headless=False)

        init_frame = preprocess_frame(env.render_frame())
        for _ in range(4):
            frame_stack.append(init_frame)
        state = np.stack(list(frame_stack), axis=0)

        done = False
        prev_coins = 0
        ep_p_reward = 0
        ep_s_reward = 0

        steps = 0
        p_action, s_action = 0, 0

        while not done:
            # Action holding (Must match the 4-frame skip logic used during training)
            if steps % 4 == 0:
                p_action = protagonist.select_action(state)
                s_action = saboteur.select_action(state)

            # Physics step
            _, _, done, info = env.step(p_action, s_action)

            # Accumulate rewards for logging
            p_rew = -10.0 if done else 0.1
            if info['coins'] > prev_coins:
                p_rew += 5.0
            s_rew = 10.0 if done else -0.1

            prev_coins = info['coins']
            ep_p_reward += p_rew
            ep_s_reward += s_rew

            # Capture next frame for the CNN
            if steps % 4 == 3 or done:
                next_frame = preprocess_frame(env.render_frame())
                frame_stack.append(next_frame)
                state = np.stack(list(frame_stack), axis=0)

            steps += 1

            # Render visually at 60 FPS
            env.render(force_fps=True)

            # Allow closing the window during eval
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        print(
            f"Eval Run {episode + 1:2d} | Score (Frames): {info['score']:4d} | Coins Collected: {info['coins']:2d} | P-Rew: {ep_p_reward:.1f} | S-Rew: {ep_s_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()