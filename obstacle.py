import pygame
import math

class Obstacle:
    def __init__(self, x, y, rotation, screen):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.height = 150
        self.width = 50
        self.screen = screen
        self.rect_color = (141,85,36)
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.image.fill(self.rect_color)

    def draw(self):
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        self.screen.blit(rotated_image, rect)
        self.move()

    def move(self):
        self.x -= 5

    def get_rect(self):
        """Return the rotated image and its rect for collision checks."""
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        return rotated_image, rect
