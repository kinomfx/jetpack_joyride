"""
game_gym.py - Updated Gym-like wrapper for CNN-based training
"""

import pygame
import numpy as np
from game import Game, GameState
# extract_state_vector is no longer strictly needed for Step 1 CNN

class GameGym:
    """
    OpenAI Gym-like environment wrapper for JetPack JoyRide
    Modified to support CNN frame capture.
    """

    def __init__(self):
        """Initialize the game environment"""
        pygame.init()
        self.game = None
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.prev_score = 0
        self.prev_coins = 0

    def render_frame(self):
        """
        Optimized frame capture for pure Pygame.
        Safely handles surface locking to prevent blit errors.
        """
        if self.game is None:
            return None

        # 1. Ensure the game is rendered
        self.game.render()

        surface = pygame.display.get_surface()

        # 2. Grab the pixels and immediately copy/close the reference
        try:
            # We use .copy() or convert to a numpy array to release the lock immediately
            pixels = pygame.surfarray.pixels3d(surface)
            frame = np.array(pixels).transpose([1, 0, 2])

            # 3. Explicitly delete the pixels3d reference to unlock the surface
            del pixels

            return frame
        except Exception as e:
            # Fallback if pixels3d is unavailable
            return pygame.surfarray.array3d(surface).transpose([1, 0, 2])

    def reset(self):
        """Reset the environment. Returns None because train.py handles the first frame capture."""
        if self.game is not None:
            pygame.event.clear()

        self.game = Game(start_state=GameState.RUNNING)
        self.prev_score = 0
        self.prev_coins = 0

        # We return None because the CNN training loop will call render_frame()
        # immediately after reset to build the initial stack.
        return None

    def step(self, action):
        """Execute one time step in the environment"""
        pygame.event.pump()

        # Execute action
        if action == 1:
            self.game.agent.move_up(self.game.upward_force)

        # Update game logic (Physics and Spawning)
        self.game.apply_gravity()

        mult = 1.0 + (self.game.score / self.game.spawner.speed_multiplier)
        mult = min(mult, 2.25)
        self.game.spawner.speed_multiplier = mult

        agent_mask = self.game.agent.get_mask()
        self.game.spawner.update_obstacles()
        collision = self.game.spawner.check_obstacle_collision(self.game.agent, agent_mask)

        self.game.spawner.update_coins()
        coin_collected = self.game.spawner.check_coin_collision(self.game.agent, agent_mask)
        if coin_collected:
            self.game.coin_count += 1

        self.game.score += 1

        # Check if episode is done
        done = collision or self.game.state != GameState.RUNNING

        # Reward logic
        reward = 0.1
        if collision:
            reward = -10.0 # Adjusted for CNN stability
        elif coin_collected:
            reward += 5.0

        info = {
            'score': self.game.score,
            'coins': self.game.coin_count,
            'collision': collision
        }

        # We return None for the 'state' because the CNN uses render_frame() instead
        return None, reward, done, info

    def render(self):
        """Display the game to the user"""
        if self.game is not None:
            self.game.render()
            self.clock.tick(self.fps)

    def close(self):
        pygame.quit()