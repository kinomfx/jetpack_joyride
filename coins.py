import pygame

class Coin:
    def __init__(self, x, y, screen):
        self.x = x
        self.y = y
        self.radius = 15
        self.screen = screen
        self.color = (255, 223, 0)  # Gold color

    def draw(self):
        pygame.draw.circle(self.screen, self.color, (self.x, self.y), self.radius)

    def move(self):
        self.x -= 5  # Move left by 5 pixels

    def get_rect(self):
        """Return the rect for collision checks."""
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)