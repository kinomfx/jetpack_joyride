import pygame
import random
from obstacle import *
from agent import *
from platform import Platform


class JetPackJoyRide:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((700, 700))
        pygame.display.set_caption("JetPack JoyRide")
        self.screen_height = 700
        self.screen_width = 700
        # agents
        self.agent = Agent(self.screen)
        # platform
        self.platform = Platform(self.screen, self.screen_width, self.screen_height)
        # Game properties
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
        self.score_pos = (self.screen_width - 25, 10)
        # Physics
        self.gravity = 5
        self.upward_force = 10  # how strong the jetpack push is
        self.obstacle_array = [Obstacle(1000, 500, 50, self.screen)]
        self.sleep_time = random.randint(140, 200)
        # self.obstacle = Obstacle(500 , 500 , 50 , self.screen)
        self.agent_mask = self.agent.get_mask()


    def draw_score(self):
        font = pygame.font.SysFont("comicsans", 30)
        text = font.render("Score: " + str(int(self.score)), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topright = self.score_pos  # visible at top-left corner
        self.screen.blit(text, text_rect)

    def draw_obstacles(self):
        if (self.sleep_time <= 0):
            rotation = random.choice([0, 22.5, 45, 67.5, 90, -22.5, -45, -67.5])
            height = random.randint(200, 600)
            self.obstacle_array.append(Obstacle(700 + 50, height, rotation, self.screen))
            self.sleep_time = random.randint(140, 200)
        while (len(self.obstacle_array) > 0 and self.obstacle_array[0].x + self.obstacle_array[0].height < 0):
            self.obstacle_array.pop(0)
        for obs in self.obstacle_array:
            obs.draw()
            rotated_image, rect = obs.get_rect()
            obstacle_mask = pygame.mask.from_surface(rotated_image)

            # Calculate offset between agent and obstacle
            offset_x = int(rect.left - (self.agent.circle_pos[0] - self.agent.agent_radius))
            offset_y = int(rect.top - (self.agent.circle_pos[1] - self.agent.agent_radius))

            if obstacle_mask.overlap(self.agent_mask, (-offset_x, -offset_y)):
                print("💥 Collision Detected!")
                self.running = False  # stop game on collision (or handle however you like)
                break

    def draw(self):
        self.screen.fill("black")
        self.platform.draw()
        # draw score text
        self.draw_score()
        # draw player
        self.agent.draw_agent()
        # obstacle drawing
        self.draw_obstacles()

    def increase_score(self):
        self.score += 1

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

            # obstacles

            # draw everything
            self.sleep_time -= 1
            self.draw()
            self.increase_score()
            pygame.display.update()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    game = JetPackJoyRide()
    game.run()