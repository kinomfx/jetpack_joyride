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

        # Zone System logic
        self.current_zone = 1  # 1 = Obstacles (Adversarial), 2 = Coins (Pattern)
        self.zone_spawn_count = 0
        self.zone_threshold = random.randint(3, 5)  # Number of obstacles before switching

    def get_obstacles(self):
        return list(self.obstacle_array)

    def get_coins(self):
        return list(self.coin_array)

    def add_obstacle(self, obstacle):
        self.obstacle_array.append(obstacle)

    def add_coin(self, coin):
        self.coin_array.append(coin)

    def can_spawn_now(self):
        """Allows spawning only if cooldown is off and in Obstacle Zone."""
        return self.spawn_cooldown <= 0 and self.current_zone == 1

    def spawn_adversarial_obstacle(self, action):
        """Action 0: Wait, 1-4: Map to Lanes[cite: 1]."""
        if action == 0 or not self.can_spawn_now():
            return

        lane_idx = action - 1
        height = self.lanes[lane_idx]

        new_obs = Obstacle(env.SCREEN_WIDTH + env.OBSTACLE_WIDTH, height, 0, False, 0)
        self.add_obstacle(new_obs)

        # Track activity for zone switching[cite: 1]
        self.zone_spawn_count += 1
        self.spawn_cooldown = 45 / max(self.speed_multiplier, 0.5)

        self._check_zone_switch()

    def _check_zone_switch(self):
        """Transition between Adversarial and Reward phases[cite: 1]."""
        if self.current_zone == 1 and self.zone_spawn_count >= self.zone_threshold:
            # Switch to Coin Zone
            self.current_zone = 2
            self.zone_spawn_count = 0
            self.spawn_random_pattern()  # Trigger coins immediately[cite: 1]

        elif self.current_zone == 2 and len(self.coin_array) == 0:
            # Switch back to Obstacle Zone once coins are cleared/gone
            self.current_zone = 1
            self.zone_threshold = random.randint(3, 5)

    def update_obstacles(self, saboteur_action=None):
        """Saboteur action is only processed in Zone 1[cite: 1]."""
        if self.current_zone == 1:
            self.spawn_adversarial_obstacle(saboteur_action)

        while len(self.obstacle_array) > 0 and self.obstacle_array[0].x + self.obstacle_array[0].width < 0:
            self.obstacle_array.pop(0)

        for obs in self.obstacle_array:
            obs.move(self.speed_multiplier)

        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= self.speed_multiplier

    def update_coins(self):
        """Cleans up coins and checks if it's time to return to obstacles[cite: 1]."""
        while len(self.coin_array) > 0 and self.coin_array[0].x + self.coin_array[0].radius < 0:
            self.coin_array.pop(0)

        for coin in self.coin_array:
            coin.move(self.speed_multiplier)

        # If in coin zone and all coins have passed, switch back[cite: 1]
        if self.current_zone == 2 and len(self.coin_array) == 0:
            self._check_zone_switch()

    # --- COIN PATTERNS (REINTEGRATED)[cite: 1] ---
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
        spacing = env.COIN_RADIUS * 4
        for i in range(count):
            self.add_coin(Coin(env.SCREEN_WIDTH + i * spacing, y))

    def pattern_coin_arc(self):
        count = 7
        base_y = random.randint(200, 450)
        spacing = env.COIN_RADIUS * 4
        amplitude = 80
        for i in range(count):
            t = i / (count - 1)
            offset = int(amplitude * (1 - (2 * t - 1) ** 2))
            self.add_coin(Coin(env.SCREEN_WIDTH + i * spacing, base_y - offset))

    def pattern_zigzag(self):
        count = 8
        y = random.randint(200, 450)
        spacing = env.COIN_RADIUS * 5
        for i in range(count):
            offset = 70 if i % 2 == 0 else -70
            self.add_coin(Coin(env.SCREEN_WIDTH + i * spacing, y + offset))

    # --- Collision and Drawing ---
    def check_obstacle_collision(self, agent, agent_mask):
        for obs in self.obstacle_array:
            rotated_image, rect = obs.get_rect()
            obstacle_mask = pygame.mask.from_surface(rotated_image)
            offset_x = int(rect.left - (agent.circle_pos[0] - agent.agent_radius))
            offset_y = int(rect.top - (agent.circle_pos[1] - agent.agent_radius))
            if obstacle_mask.overlap(agent_mask, (-offset_x, -offset_y)):
                return True
        return False

    def check_coin_collision(self, agent, agent_mask):
        for coin in self.coin_array[:]:
            coin_rect = coin.get_rect()
            coin_surf = pygame.Surface((coin.radius * 2, coin.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(coin_surf, (255, 255, 255), (coin.radius, coin.radius), coin.radius)
            coin_mask = pygame.mask.from_surface(coin_surf)
            agent_left = agent.circle_pos[0] - agent.agent_radius
            agent_top = agent.circle_pos[1] - agent.agent_radius
            offset = (int(agent_left - coin_rect.left), int(agent_top - coin_rect.top))
            if coin_mask.overlap(agent_mask, offset):
                self.coin_array.remove(coin)
                return True
        return False

    def draw_obstacles(self, screen):
        for obs in self.obstacle_array: obs.draw(screen)

    def draw_coins(self, screen):
        for coin in self.coin_array: coin.draw(screen)