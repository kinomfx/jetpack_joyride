import random
import pygame
import math
from coins import Coin
from obstacle import Obstacle
import environment as env


class Spawner:
    def __init__(self):
        self.obstacle_array = []
        self.coin_array = []
        self.spawn_cooldown = 0
        self.speed_multiplier = 1.0

        # Mapping Saboteur actions (1-4) to Y-coordinates
        self.lanes = [100, 250, 400, 550]

        # Strict Zone Management
        self.current_zone = 1  # 1 = Obstacles (Adversarial), 2 = Coins (Pattern)
        self.zone_spawn_count = 0
        self.zone_threshold = random.randint(3, 5)
        self.is_transitioning = False  # Locks the spawner until screen is clear

    def get_obstacles(self):
        return list(self.obstacle_array)

    def get_coins(self):
        return list(self.coin_array)

    def add_obstacle(self, obstacle):
        self.obstacle_array.append(obstacle)

    def add_coin(self, coin):
        self.coin_array.append(coin)

    def can_spawn_now(self):
        """Strict Check: No spawning if transitioning or in wrong zone."""
        return self.spawn_cooldown <= 0 and self.current_zone == 1 and not self.is_transitioning

    def spawn_adversarial_obstacle(self, action):
        if action == 0 or not self.can_spawn_now():
            return

        lane_idx = action - 1
        height = self.lanes[lane_idx]

        new_obs = Obstacle(env.SCREEN_WIDTH + env.OBSTACLE_WIDTH, height, 0, False, 0)
        self.add_obstacle(new_obs)

        self.zone_spawn_count += 1
        self.spawn_cooldown = 45 / max(self.speed_multiplier, 0.5)

        # Lock zone and wait for clear screen
        if self.zone_spawn_count >= self.zone_threshold:
            self.is_transitioning = True

    def update_obstacles(self, saboteur_action=None):
        if self.current_zone == 1:
            self.spawn_adversarial_obstacle(saboteur_action)

        while len(self.obstacle_array) > 0 and self.obstacle_array[0].x + self.obstacle_array[0].width < 0:
            self.obstacle_array.pop(0)

        for obs in self.obstacle_array:
            obs.move(self.speed_multiplier)

        # Move to Coins ONLY when screen is clear of Obstacles
        if self.is_transitioning and self.current_zone == 1 and len(self.obstacle_array) == 0:
            self.current_zone = 2
            self.is_transitioning = False
            self.zone_spawn_count = 0
            self.spawn_random_pattern()

        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= self.speed_multiplier

    def update_coins(self):
        while len(self.coin_array) > 0 and self.coin_array[0].x + self.coin_array[0].radius < 0:
            self.coin_array.pop(0)

        for coin in self.coin_array:
            coin.move(self.speed_multiplier)

        # Move to Obstacles ONLY when screen is clear of Coins
        if self.current_zone == 2 and len(self.coin_array) == 0:
            self.current_zone = 1
            self.zone_threshold = random.randint(3, 5)
            self.zone_spawn_count = 0

    def spawn_random_pattern(self):
        choice = random.random()
        if choice < 0.33:
            self.pattern_coin_line()
        elif choice < 0.66:
            self.pattern_coin_arc()
        else:
            self.pattern_zigzag()

    def pattern_coin_line(self):
        count = random.randint(5, 8)
        y = random.randint(150, 500)
        for i in range(count): self.add_coin(Coin(env.SCREEN_WIDTH + i * (env.COIN_RADIUS * 4), y))

    def pattern_coin_arc(self):
        count = 7
        base_y = random.randint(200, 450)
        for i in range(count):
            offset = int(80 * (1 - (2 * (i / 6) - 1) ** 2))
            self.add_coin(Coin(env.SCREEN_WIDTH + i * (env.COIN_RADIUS * 4), base_y - offset))

    def pattern_zigzag(self):
        count = 8
        y = random.randint(200, 450)
        for i in range(count):
            offset = 70 if i % 2 == 0 else -70
            self.add_coin(Coin(env.SCREEN_WIDTH + i * (env.COIN_RADIUS * 5), y + offset))

    # ==========================================
    # FPS OPTIMIZATION: REWRITTEN COLLISION MATH
    # ==========================================
    def check_obstacle_collision(self, agent, agent_mask):
        agent_rect = pygame.Rect(
            agent.circle_pos[0] - agent.agent_radius,
            agent.circle_pos[1] - agent.agent_radius,
            agent.agent_radius * 2, agent.agent_radius * 2
        )
        for obs in self.obstacle_array:
            rotated_image, rect = obs.get_rect()
            # Bounding Box Pre-Check (Bypasses expensive mask math 99% of the time)
            if not agent_rect.colliderect(rect): continue

            obstacle_mask = pygame.mask.from_surface(rotated_image)
            offset_x = int(rect.left - agent_rect.left)
            offset_y = int(rect.top - agent_rect.top)
            if obstacle_mask.overlap(agent_mask, (-offset_x, -offset_y)): return True
        return False

    def check_coin_collision(self, agent, agent_mask):
        for coin in self.coin_array[:]:
            # Euclidean Distance calculation (100x faster than masks)
            dist_x = agent.circle_pos[0] - coin.x
            dist_y = agent.circle_pos[1] - coin.y
            distance = math.hypot(dist_x, dist_y)
            if distance < (agent.agent_radius + coin.radius):
                self.coin_array.remove(coin)
                return True
        return False

    def draw_obstacles(self, screen):
        for obs in self.obstacle_array: obs.draw(screen)

    def draw_coins(self, screen):
        for coin in self.coin_array: coin.draw(screen)