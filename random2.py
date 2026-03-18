import pygame
import random
import math

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
COLOR_BG = (32, 32, 40)
COLOR_FLOOR = (26, 26, 32)
COLOR_CEILING = (26, 26, 32)
COLOR_GOLD = (255, 215, 0)
COLOR_ZAPPER = (255, 255, 0)
COLOR_WARNING = (255, 0, 0)
COLOR_TEXT = (255, 215, 0)
COLOR_WHITE = (255, 255, 255)

# --- Game Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Joyride Protocol")
clock = pygame.time.Clock()
font_large = pygame.font.SysFont("Arial", 48, bold=True)
font_small = pygame.font.SysFont("Arial", 24, bold=True)


# --- Classes ---

class Player:
    def __init__(self):
        self.width = 40
        self.height = 60
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.vy = 0
        self.gravity = 0.6
        self.lift = -12
        self.grounded = False
        self.run_frame = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, is_thrusting):
        if is_thrusting:
            self.vy -= 1.2
            if self.vy < self.lift:
                self.vy = self.lift
        else:
            self.vy += self.gravity

        self.y += self.vy

        # Floor Collision
        floor_y = SCREEN_HEIGHT - 50
        if self.y + self.height > floor_y:
            self.y = floor_y - self.height
            self.vy = 0
            self.grounded = True
        else:
            self.grounded = False

        # Ceiling Collision
        if self.y < 0:
            self.y = 0
            self.vy = 0

        # Animation state
        if self.grounded:
            self.run_frame += 0.2
        else:
            self.run_frame = 0

        self.rect.y = int(self.y)

    def draw(self, surface):
        # Draw Player Body (Barry-esque)
        # Suit
        pygame.draw.rect(surface, (51, 51, 51), (self.x + 5, self.y + 20, 25, 30))
        # Head
        pygame.draw.rect(surface, (255, 204, 170), (self.x + 5, self.y, 25, 20))  # Skin
        pygame.draw.rect(surface, (68, 34, 0), (self.x + 5, self.y, 25, 8))  # Hair
        # Jetpack
        pygame.draw.rect(surface, (136, 136, 136), (self.x - 5, self.y + 20, 15, 25))
        pygame.draw.rect(surface, (68, 68, 68), (self.x, self.y + 40, 5, 10))  # Thruster

        # Legs
        if self.grounded:
            leg_offset = math.sin(self.run_frame) * 10
            pygame.draw.rect(surface, (34, 34, 34), (self.x + 10 + leg_offset, self.y + 50, 8, 10))
            pygame.draw.rect(surface, (34, 34, 34), (self.x + 15 - leg_offset, self.y + 50, 8, 10))
        else:
            pygame.draw.rect(surface, (34, 34, 34), (self.x + 10, self.y + 50, 8, 8))
            pygame.draw.rect(surface, (34, 34, 34), (self.x + 20, self.y + 52, 8, 8))

        # Arm
        pygame.draw.rect(surface, (255, 255, 255), (self.x + 15, self.y + 25, 15, 8))


class Particle:
    def __init__(self, x, y, p_type):
        self.x = x
        self.y = y
        self.type = p_type
        self.size = random.randint(2, 7)
        self.speed_x = (random.random() - 0.5) * 4 - 2
        self.speed_y = (random.random() - 0.5) * 4 + 2
        self.life = 255

        if p_type == 'zap':
            self.color = (255, 255, 0)
            self.speed_x = (random.random() - 0.5) * 10
            self.speed_y = (random.random() - 0.5) * 10
            self.decay = 15
        elif p_type == 'coin':
            self.color = (255, 215, 0)
            self.speed_y = -2
            self.decay = 5
        else:  # fire/smoke
            r = 255
            g = random.randint(0, 150)
            b = 0
            self.color = (r, g, b)
            self.decay = 5

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.life -= self.decay

    def draw(self, surface):
        if self.life > 0:
            s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            color_with_alpha = (*self.color, int(self.life))
            pygame.draw.circle(s, color_with_alpha, (self.size, self.size), self.size)
            surface.blit(s, (self.x - self.size, self.y - self.size))


class Obstacle:
    def __init__(self, game_speed):
        self.x = SCREEN_WIDTH + 100
        self.type = 'ZAPPER' if random.random() > 0.5 else 'MISSILE_WARNING'
        self.active = True
        self.game_speed = game_speed

        if self.type == 'ZAPPER':
            self.width = 20
            self.height = random.randint(100, 200)
            self.y = random.randint(0, SCREEN_HEIGHT - 100 - self.height)
            self.rect = pygame.Rect(self.x, self.y + 10, self.width, self.height - 20)  # Hitbox slightly smaller
        else:
            # Missile
            self.width = 60
            self.height = 30
            self.y = random.randint(50, SCREEN_HEIGHT - 150)
            self.warning_timer = 60
            self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, player_y, game_speed, particles):
        if self.type == 'MISSILE_WARNING':
            if self.warning_timer > 0:
                self.warning_timer -= 1
                # Track player roughly
                if player_y > self.y:
                    self.y += 2
                else:
                    self.y -= 2
            else:
                self.type = 'MISSILE'
                self.speed = game_speed * 2.5
        elif self.type == 'MISSILE':
            self.x -= self.speed
            self.rect.x = int(self.x)
            self.rect.y = int(self.y)
            # Smoke trail
            particles.append(Particle(self.x + self.width, self.y + self.height // 2, 'smoke'))
        else:  # Zapper
            self.x -= game_speed
            self.rect.x = int(self.x)
            self.rect.y = int(self.y)

        if self.x + self.width < -50:
            self.active = False

    def draw(self, surface, frame_count):
        if self.type == 'ZAPPER':
            # Draw end caps
            pygame.draw.circle(surface, (136, 136, 136), (int(self.x + 10), int(self.y)), 10)
            pygame.draw.circle(surface, (136, 136, 136), (int(self.x + 10), int(self.y + self.height)), 10)

            # Electric field (jagged line)
            points = []
            for i in range(0, self.height, 10):
                offset_x = random.randint(-5, 5)
                points.append((self.x + 10 + offset_x, self.y + i))
            if len(points) > 1:
                pygame.draw.lines(surface, COLOR_ZAPPER, False, points, 3)

        elif self.type == 'MISSILE_WARNING':
            if (frame_count // 10) % 2 == 0:
                # Warning Icon
                text = font_small.render("!", True, COLOR_WARNING)
                surface.blit(text, (SCREEN_WIDTH - 50, self.y))
                # Target line
                pygame.draw.line(surface, (100, 0, 0), (SCREEN_WIDTH - 50, self.y + 15), (0, self.y + 15), 1)

        elif self.type == 'MISSILE':
            # Rocket Body
            pygame.draw.rect(surface, (204, 0, 0), (self.x, self.y, self.width, self.height))
            # Fins
            pygame.draw.rect(surface, (85, 0, 0), (self.x + 40, self.y - 5, 5, self.height + 10))
            # Nose cone (triangle approx)
            pygame.draw.polygon(surface, (204, 0, 0), [(self.x, self.y), (self.x, self.y + self.height),
                                                       (self.x - 15, self.y + self.height // 2)])


class Coin:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 15
        self.active = True
        self.rect = pygame.Rect(x - 15, y - 15, 30, 30)

    def update(self, game_speed):
        self.x -= game_speed
        self.rect.x = int(self.x - self.size)

        if self.x < -20:
            self.active = False

    def draw(self, surface, frame_count):
        # Spin effect (scaling width)
        scale_x = abs(math.sin(frame_count * 0.1))
        width = int(self.size * scale_x)
        if width < 1: width = 1

        rect = pygame.Rect(self.x - width, self.y - self.size, width * 2, self.size * 2)
        pygame.draw.ellipse(surface, COLOR_GOLD, rect)
        pygame.draw.ellipse(surface, (180, 120, 0), rect, 2)


# --- Main Game Loop ---

def main():
    running = True
    game_state = "START"  # START, PLAYING, GAMEOVER

    player = Player()
    particles = []
    obstacles = []
    coins = []

    score = 0
    distance = 0
    game_speed = 5
    frame_count = 0

    # Background scrolling var
    wall_x = 0

    def reset_game():
        nonlocal player, particles, obstacles, coins, score, distance, game_speed, frame_count
        player = Player()
        particles = []
        obstacles = []
        coins = []
        score = 0
        distance = 0
        game_speed = 6
        frame_count = 0

    while running:
        clock.tick(FPS)

        # --- Input Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                if game_state == "START":
                    game_state = "PLAYING"
                    reset_game()
                elif game_state == "GAMEOVER":
                    game_state = "PLAYING"
                    reset_game()

        keys = pygame.key.get_pressed()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        is_thrusting = keys[pygame.K_SPACE] or mouse_pressed

        # --- Updates ---

        if game_state == "PLAYING":
            frame_count += 1
            distance += game_speed / 10
            if frame_count % 600 == 0:
                game_speed += 0.5

            # Player
            player.update(is_thrusting)
            if is_thrusting and frame_count % 2 == 0:
                particles.append(Particle(player.x + 10, player.y + player.height - 10, 'fire'))

            # Spawn Obstacles
            if frame_count % 120 == 0:
                obstacles.append(Obstacle(game_speed))

            # Spawn Coins
            if frame_count % 60 == 0 and random.random() > 0.5:
                start_y = random.randint(50, SCREEN_HEIGHT - 150)
                for i in range(5):
                    c_x = SCREEN_WIDTH + (i * 40)
                    c_y = start_y + math.sin(i) * 30
                    coins.append(Coin(c_x, c_y))

            # Update Entities
            for p in particles: p.update()
            for o in obstacles: o.update(player.y, game_speed, particles)
            for c in coins: c.update(game_speed)

            # Cleanup
            particles = [p for p in particles if p.life > 0]
            obstacles = [o for o in obstacles if o.active]
            coins = [c for c in coins if c.active]

            # Collisions
            player_rect = pygame.Rect(player.x + 5, player.y + 5, player.width - 10, player.height - 10)

            # Coin Collision
            for c in coins:
                if c.active and player_rect.colliderect(c.rect):
                    c.active = False
                    score += 1
                    for _ in range(5):
                        particles.append(Particle(c.x, c.y, 'coin'))

            # Obstacle Collision
            for o in obstacles:
                if o.active and o.type != 'MISSILE_WARNING':
                    if player_rect.colliderect(o.rect):
                        game_state = "GAMEOVER"
                        for _ in range(30):
                            particles.append(Particle(player.x + 20, player.y + 30, 'zap'))

        elif game_state == "GAMEOVER":
            # Still animate particles
            for p in particles: p.update()
            particles = [p for p in particles if p.life > 0]

        # --- Drawing ---

        # Background
        wall_x -= game_speed * 0.5
        if wall_x <= -200: wall_x = 0

        screen.fill(COLOR_BG)
        # Ceiling & Floor
        pygame.draw.rect(screen, COLOR_CEILING, (0, 0, SCREEN_WIDTH, 50))
        pygame.draw.rect(screen, COLOR_FLOOR, (0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, 50))
        pygame.draw.rect(screen, (51, 51, 51), (0, SCREEN_HEIGHT - 45, SCREEN_WIDTH, 5))

        # Lab Pillars
        for i in range(0, SCREEN_WIDTH + 200, 200):
            x_pos = (i + wall_x) % (SCREEN_WIDTH + 200)
            if x_pos < SCREEN_WIDTH:
                pygame.draw.rect(screen, (32, 32, 40), (x_pos, 50, 20, SCREEN_HEIGHT - 100))

        # Draw Entities
        player.draw(screen)
        for c in coins: c.draw(screen, frame_count)
        for o in obstacles: o.draw(screen, frame_count)
        for p in particles: p.draw(screen)

        # UI
        score_text = font_small.render(f"COINS: {score}", True, COLOR_GOLD)
        dist_text = font_small.render(f"{int(distance)} M", True, COLOR_WHITE)
        screen.blit(score_text, (20, 20))
        screen.blit(dist_text, (SCREEN_WIDTH - 100, 20))

        if game_state == "START":
            # Dark overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            title = font_large.render("JOYRIDE PROTOCOL", True, COLOR_GOLD)
            sub = font_small.render("Press SPACE to Fly", True, COLOR_WHITE)

            screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            screen.blit(sub, (SCREEN_WIDTH // 2 - sub.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

        elif game_state == "GAMEOVER":
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            msg = font_large.render("MISSION FAILED", True, COLOR_WHITE)
            stats = font_small.render(f"Distance: {int(distance)}m  |  Coins: {score}", True, COLOR_GOLD)
            retry = font_small.render("Press SPACE to Try Again", True, COLOR_WHITE)

            screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            screen.blit(stats, (SCREEN_WIDTH // 2 - stats.get_width() // 2, SCREEN_HEIGHT // 2 + 10))
            screen.blit(retry, (SCREEN_WIDTH // 2 - retry.get_width() // 2, SCREEN_HEIGHT // 2 + 50))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()