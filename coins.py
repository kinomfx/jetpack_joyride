import pygame
import environment as env

class Coin:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = env.COIN_RADIUS
        self.color = env.COIN_COLOR

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

    def move(self, speed_multiplier=1.0):
        # move left scaled by speed multiplier
        self.x -= env.COIN_SPEED * speed_multiplier

    def get_rect(self):
        """Return the rect for collision checks."""
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)