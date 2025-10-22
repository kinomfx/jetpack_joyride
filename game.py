import pygame
from obstacle import *
from agent import *
from platform import *
class JetPackJoyRide:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((700 , 700))
        pygame.display.set_caption("JetPack JoyRide")
        self.screen_height = 700
        self.screen_width = 700
        #agents
        self.agent = Agent(self.screen)
        #platform
        self.platform = Platform(self.screen , self.screen_width, self.screen_height)
        # Game properties
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
        self.score_pos = (self.screen_width-25 , 10)
        # Physics
        self.gravity = 5
        self.upward_force = 10  # how strong the jetpack push is
    def draw_score(self):
        font = pygame.font.SysFont("comicsans", 30)
        text = font.render("Score: " + str(int(self.score // 10)), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topright = self.score_pos  # visible at top-left corner
        self.screen.blit(text, text_rect)

    def draw(self):
        self.screen.fill("black")
        self.platform.draw()
        # draw score text
        self.draw_score()
        # draw player
        self.agent.draw_agent()
    def increase_score(self):
        self.score += 0.1

    def apply_gravity(self):
        if self.agent.circle_pos[1] + self.agent.agent_radius < self.platform.rect.top:
            self.agent.circle_pos[1] += self.gravity

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # check continuous key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.agent.move_up(self.upward_force)

            # apply gravity (fall down when not pressing space)
            self.apply_gravity()

            # draw everything
            self.draw()
            self.increase_score()
            pygame.display.update()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    game = JetPackJoyRide()
    game.run()
