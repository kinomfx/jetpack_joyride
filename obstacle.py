import pygame
import math
import environment as env

class Obstacle:
    def __init__(self, x, y, rotation, rotating=False, rotation_speed=0):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.rotating = rotating
        self.rotation_speed = rotation_speed
        self.height = env.OBSTACLE_HEIGHT
        self.width = env.OBSTACLE_WIDTH
        self.rect_color = env.OBSTACLE_COLOR
        self.image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.image.fill(self.rect_color)

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, rect)

    def move(self, speed_multiplier=1.0):
        # move left scaled by speed multiplier
        self.x -= env.OBSTACLE_SPEED * speed_multiplier
        if self.rotating:
            self.rotation = (self.rotation + self.rotation_speed) % 360

    def get_rect(self):
        """Return the rotated image and its rect for collision checks."""
        rotated_image = pygame.transform.rotate(self.image, self.rotation)
        rect = rotated_image.get_rect(center=(self.x, self.y))
        return rotated_image, rect
