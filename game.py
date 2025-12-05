import pygame
from enum import Enum, auto
from agent import Agent
from platform import Platform
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
    def __init__(self, start_state=GameState.MENU):
        pygame.init()
        pygame.event.clear()  # Clear event queue to avoid leftover events
        self.screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("JetPack JoyRide")
        self.screen_width = env.SCREEN_WIDTH
        self.screen_height = env.SCREEN_HEIGHT
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

    def apply_gravity(self):
        if self.agent.circle_pos[1] + self.agent.agent_radius < self.platform.rect.top:
            self.agent.circle_pos[1] += self.gravity

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = GameState.QUIT

    def update(self):
        if self.state != GameState.RUNNING:
            return
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

    def render(self):
        self.screen.fill(env.BACKGROUND_COLOR)
        self.platform.draw(self.screen)
        self.agent.draw_agent(self.screen)
        self.spawner.draw_obstacles(self.screen)
        self.spawner.draw_coins(self.screen)
        self.hud.draw(self.score, self.coin_count)
        pygame.display.update()

    def run(self):
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
        """Main game loop."""
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
        # Save score when game ends
        self.score_manager.add_score(self.score, self.coin_count)
        while waiting:
            self.hud.draw_endscreen()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.state = GameState.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Only option: go to menu
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
                        self.state = GameState.RUNNING
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