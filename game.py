import pygame
from enum import Enum, auto
from agent import Agent
from platform import Platform
from hud import HUD
from spawner import Spawner
from score_manager import ScoreManager
import environment as env
#remove
from state_extractor import extract_state, extract_state_vector
import pprint
#remove

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
        # state print debug timer (ms)
        self._last_state_print = pygame.time.get_ticks()
        # agent vertical velocity estimate (pixels per frame)
        self._agent_vy = 0.0

    def reset_game(self):
        """Reset game state for a restart without full reinitialization."""
        pygame.event.clear()
        self.score = 0
        self.coin_count = 0
        self.agent = Agent()
        self.spawner = Spawner()
        self.state = GameState.RUNNING
        # reset state print timer so immediate print doesn't occur
        self._last_state_print = pygame.time.get_ticks()
        # reset agent velocity
        self._agent_vy = 0.0

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
        # update speed multiplier based on score
        mult = 1.0 + (self.score / env.SPEED_SCORE_DIVISOR)
        if mult > env.MAX_SPEED_MULTIPLIER:
            mult = env.MAX_SPEED_MULTIPLIER
        # inform spawner about current speed multiplier
        self.spawner.speed_multiplier = mult
        # estimate agent vertical velocity: sample position before update
        prev_y = self.agent.circle_pos[1]
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
        # update agent vertical velocity estimate (positive = moving up)
        self._agent_vy = float(prev_y - self.agent.circle_pos[1])

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
                #remove
                # print the RL state every 5 seconds for debugging
                try:
                    now = pygame.time.get_ticks()
                    if now - self._last_state_print >= 5000:
                        state = extract_state(self)
                        print("[StateExtractor]", end=" ")
                        pprint.pprint(state)
                        # also print normalized vector state (if numpy available)
                        try:
                            vec = extract_state_vector(self)
                            try:
                                import numpy as _np
                                print("[StateVector]", _np.array2string(vec, precision=3))
                            except Exception:
                                # fallback: print as list
                                print("[StateVector]", list(vec))
                        except Exception as e:
                            print(f"[StateVector] unavailable: {e}")
                        self._last_state_print = now
                except Exception as e:
                    print(f"[StateExtractor] error: {e}")
                #remove
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
                        # ESC goes to menu
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
                        # Reset game state before starting a new game
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