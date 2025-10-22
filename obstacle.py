import pygame
import math

class Obstacle:
    def __init__(self, x, y, rotation, screen):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.height = 200
        self.width = 100
        self.screen = screen
        self.rect_color = (169, 169, 169)

        # Create a surface for the rectangle
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.image.fill(self.rect_color)

    def draw(self):
        # Rotate the surface
        rotated_image = pygame.transform.rotate(self.image, self.rotation)

        # Get the new rect (centered properly)
        rect = rotated_image.get_rect(center=(self.x, self.y))

        # Draw it
        self.screen.blit(rotated_image, rect)

    def move(self):
        self.y -= 5

