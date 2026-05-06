import pygame
from enum import Enum, auto
from agent import Agent
from Platform import *
from hud import HUD
from spawner import Spawner
from score_manager import ScoreManager
import environment as env


class GameState(Enum):
    MENU = auto()
    RUNNING = auto()
    GAME_OVER = auto()
    HIGHSCORES = auto()
    QUIT = auto()


class Game:
    def __init__(self, start_state=GameState.MENU, headless=True):
        pygame.init()
        self.headless = headless
        self.screen_width = env.SCREEN_WIDTH
        self.screen_height = env.SCREEN_HEIGHT

        if not self.headless:
            # Set the actual window
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("JetPack JoyRide")
        else:
            # Create the virtual surface
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.clock = pygame.time.Clock()
        self.state = start_state
        self.score = 0
        self.coin_count = 0
        self.gravity = env.GRAVITY
        self.upward_force = env.UPWARD_FORCE
        self.agent = Agent()
        self.platform = Platform(self.screen_width, self.screen_height)
        self.hud = HUD(self.screen, self.screen_width, self.screen_height)
        self.spawner = Spawner()
        self.score_manager = ScoreManager()
        self._last_state_print = pygame.time.get_ticks()
        self._agent_vy = 0.0

    def reset_game(self, headless=None):
        """Reset game state for a restart without full reinitialization."""
        pygame.event.clear()

        # Allow updating headless state on reset
        if headless is not None:
            self.headless = headless
            if not self.headless:
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.hud.screen = self.screen  # Update HUD reference

        self.score = 0
        self.coin_count = 0
        self.agent = Agent()
        self.spawner = Spawner()
        self.state = GameState.RUNNING
        self._last_state_print = pygame.time.get_ticks()
        self._agent_vy = 0.0

    def apply_gravity(self):
        if self.agent.circle_pos[1] + self.agent.agent_radius < self.platform.rect.top:
            self.agent.circle_pos[1] += self.gravity

    def handle_events(self):
        # Only process keyboard/window events if there is a window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = GameState.QUIT

    def update(self, action=None):
        """
        Update game state.
        :param action: If provided (RL mode), 1 = Jump, 0 = Nothing.
        """
        if self.state != GameState.RUNNING:
            return

        mult = 1.0 + (self.score / env.SPEED_SCORE_DIVISOR)
        if mult > env.MAX_SPEED_MULTIPLIER:
            mult = env.MAX_SPEED_MULTIPLIER
        self.spawner.speed_multiplier = mult

        prev_y = self.agent.circle_pos[1]

        # Support for both manual play and RL action inputs
        if action is not None:
            if action == 1:
                self.agent.move_up(self.upward_force)
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.agent.move_up(self.upward_force)

        self.apply_gravity()
        agent_mask = self.agent.get_mask()
        self.spawner.update_obstacles()

        if self.spawner.check_obstacle_collision(self.agent, agent_mask):
            self.state = GameState.GAME_OVER
            return

        self.spawner.update_coins()
        if self.spawner.check_coin_collision(self.agent, agent_mask):
            self.coin_count += 1

        self.score += 1
        self._agent_vy = float(prev_y - self.agent.circle_pos[1])

    def render(self):
        """Draws everything to the internal self.screen."""
        self.screen.fill(env.BACKGROUND_COLOR)
        self.platform.draw(self.screen)
        self.agent.draw_agent(self.screen)
        self.spawner.draw_obstacles(self.screen)
        self.spawner.draw_coins(self.screen)
        self.hud.draw(self.score, self.coin_count)

    def run(self):
        """Standard manual play loop."""
        while self.state != GameState.QUIT:
            if self.state == GameState.MENU:
                self.show_startscreen()
            elif self.state == GameState.HIGHSCORES:
                self.show_highscores()
            elif self.state == GameState.RUNNING:
                self.run_game()
            elif self.state == GameState.GAME_OVER:
                self.show_endscreen()
        pygame.quit()

    def run_game(self):
        while self.state == GameState.RUNNING:
            self.handle_events()
            if self.state == GameState.RUNNING:
                self.update()
                self.render()
            self.clock.tick(60)
        if self.state == GameState.RUNNING:
            self.state = GameState.GAME_OVER

    def show_endscreen(self):
        waiting = True
        clock = pygame.time.Clock()
        self.score_manager.add_score(self.score, self.coin_count)
        while waiting:
            self.hud.draw_endscreen()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.state = GameState.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                        self.state = GameState.MENU
            clock.tick(60)

    def show_startscreen(self):
        waiting = True
        clock = pygame.time.Clock()
        while waiting:
            self.hud.draw_startscreen()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.state = GameState.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                        self.reset_game()
                    elif event.key == pygame.K_h:
                        waiting = False
                        self.state = GameState.HIGHSCORES
                    elif event.key == pygame.K_ESCAPE:
                        waiting = False
                        self.state = GameState.QUIT
            clock.tick(60)

    def show_highscores(self):
        waiting = True
        clock = pygame.time.Clock()
        while waiting:
            self.hud.draw_highscores()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.state = GameState.QUIT
                if event.type == pygame.KEYDOWN:
                    waiting = False
                    self.state = GameState.MENU
            clock.tick(60)


if __name__ == "__main__":
    game = Game()
    game.run()