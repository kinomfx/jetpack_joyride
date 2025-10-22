import pygame
class Agent:
    def __init__(self,  screen):
        self.circle_pos = [100, 350]
        self.agent_radius = 30
        self.screen = screen
    def draw_agent(self):
        pygame.draw.circle(self.screen, "red", self.circle_pos, self.agent_radius)
    def move_up(self , upward_force):
        if self.circle_pos[1] - self.agent_radius >= 0:
            self.circle_pos[1] -= upward_force