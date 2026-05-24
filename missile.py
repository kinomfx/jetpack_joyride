import pygame
import environment as env


class Missile:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 60
        self.height = 30
        self.color = (255, 50, 50)  # Red missile
        self.warning_color = (255, 165, 0)  # Orange warning
        self.warning_timer = 60  # Wait 60 frames (1 second) before firing
        self.speed = env.OBSTACLE_SPEED * 3.5  # Much faster than obstacles
        self.rect = pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

    def move(self, speed_multiplier=1.0):
        # Count down the warning phase
        if self.warning_timer > 0:
            self.warning_timer -= max(1, speed_multiplier)
        else:
            # Fire phase
            self.x -= self.speed * speed_multiplier

        self.rect.center = (self.x, self.y)

    def draw(self, screen):
        if self.warning_timer > 0:
            # Draw a flashing warning square at the right edge of the screen
            warn_rect = pygame.Rect(env.SCREEN_WIDTH - 40, self.y - 15, 30, 30)
            color = self.warning_color if int(self.warning_timer) % 10 > 5 else (255, 0, 0)
            pygame.draw.rect(screen, color, warn_rect)

            # Draw an exclamation mark inside it
            pygame.draw.rect(screen, (0, 0, 0), (env.SCREEN_WIDTH - 28, self.y - 10, 6, 14))
            pygame.draw.rect(screen, (0, 0, 0), (env.SCREEN_WIDTH - 28, self.y + 6, 6, 6))
        else:
            # Draw the actual missile
            pygame.draw.rect(screen, self.color, self.rect)
            # Add a little engine exhaust flame for visuals
            pygame.draw.polygon(screen, (255, 200, 0), [
                (self.rect.right, self.rect.centery),
                (self.rect.right + 20, self.rect.centery - 10),
                (self.rect.right + 20, self.rect.centery + 10)
            ])

    def get_rect(self):
        return self.rect