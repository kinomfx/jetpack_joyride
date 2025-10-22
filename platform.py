import pygame
class Platform:
    def __init__(self, screen , screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.rect = pygame.Rect(0, self.screen_width - 100, self.screen_width, 100)
        self.rect_color = (169, 169, 169)
    def draw(self):
        pygame.draw.rect(self.screen, self.rect_color, self.rect)