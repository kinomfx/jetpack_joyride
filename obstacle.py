import pygame
class Obstacle :
    def __init__(self , x , y, rotation , screen):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.height = 200
        self.width = 100
        self.screen = screen
        self.ob = pygame.Rect(self.x, self.y, self.width, self.height)
        self.rect_color = (169, 169, 169)
    def draw(self):
        self.ob = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(self.screen, self.rect_color, self.ob)
    def move(self):
        self.x -= 5


