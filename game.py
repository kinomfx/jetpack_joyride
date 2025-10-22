import pygame
from obstacle import *
from agent import *
from platform import *
from obstacle import *
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

            #obstacles
            #here we need to add obstacles and check for collision there are 2 surfaces
            #one is the agent and the other is the Obstacle class
            #which btw will be stored in an array of obstacles
            #there should be a probability function like 90% probability that an obstacle will not be created at 120 times a second so 1/10 * 120 is still 12 obstacles persecond which is alot we just need 2 obstacles per the whole screen time
            #this is the current thing that i have to figure out
            # draw everything
            self.draw()
            self.increase_score()
            pygame.display.update()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    game = JetPackJoyRide()
    game.run()
