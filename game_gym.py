import pygame
import numpy as np
from game import Game, GameState


class GameGym:
    def __init__(self):
        pygame.init()
        self.game = None
        self.clock = pygame.time.Clock()
        self.fps = 60

    def reset(self, headless=True):
        """Reset the environment. Headless=True hides the window."""
        if self.game is not None:
            pygame.display.quit()
            pygame.display.init()
            pygame.event.clear()

        # Initialize the game with the headless flag
        self.game = Game(start_state=GameState.RUNNING, headless=headless)
        return None

    def step(self, p_action, s_action):
        """
        Modified step function to handle both agents.
        p_action: Action for the Protagonist (Player)
        s_action: Action for the Saboteur (Obstacle Spawner)
        """
        pygame.event.pump()

        # 1. Apply Protagonist Action (Physics)
        if p_action == 1:
            self.game.agent.move_up(self.game.upward_force)

        self.game.apply_gravity()

        # 2. Apply Saboteur Action to the Spawner
        # This passes the adversarial decision to the new update_obstacles method
        self.game.spawner.update_obstacles(saboteur_action=s_action)

        # 3. Standard Environment Updates
        self.game.spawner.update_coins()

        # 4. Check for Adversarial Collision
        collision = self.game.spawner.check_obstacle_collision(
            self.game.agent, self.game.agent.get_mask()
        )

        # 5. Check for Protagonist Reward (Coins)
        if self.game.spawner.check_coin_collision(self.game.agent, self.game.agent.get_mask()):
            self.game.coin_count += 1

        self.game.score += 1
        done = collision or self.game.state != GameState.RUNNING

        info = {'score': self.game.score, 'coins': self.game.coin_count}

        # We return 0 for reward here because reward calculation is handled
        # in the train.py loop to facilitate zero-sum logic.
        return None, 0, done, info

    def render_frame(self):
        """Captures the frame for CNN processing."""
        self.game.render()
        # Grabs from the surface (works in headless mode too)
        surface = self.game.screen
        pixels = pygame.surfarray.pixels3d(surface)
        frame = np.array(pixels).transpose([1, 0, 2])
        del pixels  # Avoid memory leaks
        return frame

    def render(self, force_fps=True):
        """Physical rendering of the window."""
        if self.game and not self.game.headless:
            self.game.render()
            pygame.display.update()
            if force_fps:
                self.clock.tick(self.fps)

    def close(self):
        pygame.quit()