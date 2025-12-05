import pygame
import environment as env
class Platform:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.rect = pygame.Rect(0, self.screen_width - env.PLATFORM_HEIGHT, self.screen_width, env.PLATFORM_HEIGHT)
        self.rect_color = env.PLATFORM_COLOR

    def draw(self, screen):
        pygame.draw.rect(screen, self.rect_color, self.rect)