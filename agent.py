import pygame
import environment as env
class Agent:
    def __init__(self):
        self.circle_pos = env.AGENT_START_POS.copy()
        self.agent_radius = env.AGENT_RADIUS

    def draw_agent(self, screen):
        pygame.draw.circle(screen, env.AGENT_COLOR, self.circle_pos, self.agent_radius)

    def move_up(self, upward_force):
        if self.circle_pos[1] - self.agent_radius >= 0:
            self.circle_pos[1] -= upward_force

    def get_mask(self):
        surf = pygame.Surface((self.agent_radius * 2, self.agent_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 0, 0), (self.agent_radius, self.agent_radius), self.agent_radius)
        return pygame.mask.from_surface(surf)
