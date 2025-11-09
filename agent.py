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
    def get_mask(self):
        surf = pygame.Surface((self.agent_radius * 2, self.agent_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 0, 0), (self.agent_radius, self.agent_radius), self.agent_radius)
        return pygame.mask.from_surface(surf)
