import pygame

class JetPackJoyRide:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((500, 500))
        pygame.display.set_caption("JetPack JoyRide")

        # Game properties
        self.circle_pos = [100, 350]
        self.agent_radius = 40
        self.rect = pygame.Rect(0, 400, 500, 100)
        self.rect_color = (169, 169, 169)
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
        self.score_pos = (475 , 10)
        # Physics
        self.gravity = 5
        self.upward_force = 10  # how strong the jetpack push is

    def draw_agent(self):
        pygame.draw.circle(self.screen, "red", self.circle_pos, self.agent_radius)

    def draw(self):
        self.screen.fill("black")
        pygame.draw.rect(self.screen, self.rect_color, self.rect)

        # draw score text
        font = pygame.font.SysFont("comicsans", 30)
        text = font.render("Score: " + str(int(self.score//10)), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topright = self.score_pos  # visible at top-left corner
        self.screen.blit(text, text_rect)
        # draw player
        self.draw_agent()
    def increase_score(self):
        self.score += 0.1;
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # check continuous key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.circle_pos[1] -= self.upward_force  # move up while space is held

            # apply gravity (fall down when not pressing space)
            if self.circle_pos[1] + self.agent_radius < self.rect.top:
                self.circle_pos[1] += self.gravity

            # draw everything
            self.draw()
            self.increase_score()
            pygame.display.update()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    game = JetPackJoyRide()
    game.run()
