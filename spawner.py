import random
import pygame
import math
from coins import Coin
from obstacle import Obstacle
from missile import Missile
import environment as env


class Spawner:
    def __init__(self):
        self.obstacle_array = []
        self.missile_array = []
        self.coin_array = []
        self.spawn_cooldown = 0
        self.speed_multiplier = 1.0

        self.lanes = [100, 230, 360, 490]

        # 3-Phase Strict Zone Management
        self.current_zone = 1  # 1=Obstacles, 2=Missiles, 3=Coins
        self.zone_spawn_count = 0
        self.zone_threshold = random.randint(3, 5)
        self.is_transitioning = False

        # ANTI-LAZINESS TIMER
        self.lazy_timer = 0

    def get_obstacles(self):
        return list(self.obstacle_array)

    def get_missiles(self):
        return list(self.missile_array)

    def get_coins(self):
        return list(self.coin_array)

    def can_spawn_now(self):
        return self.spawn_cooldown <= 0 and not self.is_transitioning

    def _check_zone_switch(self):
        """Forces the screen to be 100% empty before switching phases."""
        if not self.is_transitioning:
            return

        if self.current_zone == 1 and len(self.obstacle_array) == 0:
            self.current_zone = 2  # Switch to Missiles
            self.is_transitioning = False
            self.zone_spawn_count = 0
            self.zone_threshold = random.randint(1, 3)
            self.spawn_cooldown = 30

        elif self.current_zone == 2 and len(self.missile_array) == 0:
            self.current_zone = 3  # Switch to Coins
            self.is_transitioning = False
            self.zone_spawn_count = 0
            self.spawn_random_pattern()

        elif self.current_zone == 3 and len(self.coin_array) == 0:
            self.current_zone = 1  # Back to Obstacles
            self.is_transitioning = False
            self.zone_spawn_count = 0
            self.zone_threshold = random.randint(3, 5)
            self.spawn_cooldown = 30

    # --- SPAWN LOGIC ---
    def spawn_adversarial_threat(self, action):
        if not self.can_spawn_now():
            return

        # MANUAL PLAY FALLBACK (If you run python game.py)
        if action is None:
            action = random.randint(1, 4)

        # RL LAZY SABOTEUR FIX
        if action == 0:
            self.lazy_timer += 1
            if self.lazy_timer >= 60:
                action = random.randint(1, 4)  # Force fire after 1 second
                self.lazy_timer = 0
            else:
                return
        else:
            self.lazy_timer = 0  # Reset timer if it chose to shoot

        lane_idx = action - 1
        height = self.lanes[lane_idx]

        if self.current_zone == 1:
            new_obs = Obstacle(env.SCREEN_WIDTH + env.OBSTACLE_WIDTH, height, 0, False, 0)
            self.obstacle_array.append(new_obs)
            self.spawn_cooldown = 45 / max(self.speed_multiplier, 0.5)
            self.zone_spawn_count += 1

        elif self.current_zone == 2:
            new_mis = Missile(env.SCREEN_WIDTH + 30, height)
            self.missile_array.append(new_mis)
            self.spawn_cooldown = 90 / max(self.speed_multiplier, 0.5)
            self.zone_spawn_count += 1

        if self.zone_spawn_count >= self.zone_threshold:
            self.is_transitioning = True

    # --- UPDATE LOOPS ---
    def update_threats(self, saboteur_action=None):
        if self.current_zone in [1, 2]:
            self.spawn_adversarial_threat(saboteur_action)

        # Update Obstacles
        while len(self.obstacle_array) > 0 and self.obstacle_array[0].x + self.obstacle_array[0].width < 0:
            self.obstacle_array.pop(0)
        for obs in self.obstacle_array:
            obs.move(self.speed_multiplier)

        # Update Missiles
        while len(self.missile_array) > 0 and self.missile_array[0].x + self.missile_array[0].width < 0:
            self.missile_array.pop(0)
        for mis in self.missile_array:
            mis.move(self.speed_multiplier)

        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= self.speed_multiplier

        self._check_zone_switch()

    def update_coins(self):
        while len(self.coin_array) > 0 and self.coin_array[0].x + self.coin_array[0].radius < 0:
            self.coin_array.pop(0)
        for coin in self.coin_array:
            coin.move(self.speed_multiplier)

        if self.current_zone == 3 and len(self.coin_array) == 0:
            self.is_transitioning = True
            self._check_zone_switch()

    # --- COIN PATTERNS ---
    def spawn_random_pattern(self):
        choice = random.random()
        if choice < 0.33:
            self.pattern_coin_line()
        elif choice < 0.66:
            self.pattern_coin_arc()
        else:
            self.pattern_zigzag()

    def pattern_coin_line(self):
        for i in range(random.randint(5, 8)):
            self.coin_array.append(Coin(env.SCREEN_WIDTH + i * (env.COIN_RADIUS * 4), random.randint(150, 450)))

    def pattern_coin_arc(self):
        base_y = random.randint(200, 450)
        for i in range(7):
            offset = int(80 * (1 - (2 * (i / 6) - 1) ** 2))
            self.coin_array.append(Coin(env.SCREEN_WIDTH + i * (env.COIN_RADIUS * 4), base_y - offset))

    def pattern_zigzag(self):
        y = random.randint(200, 450)
        for i in range(8):
            offset = 70 if i % 2 == 0 else -70
            self.coin_array.append(Coin(env.SCREEN_WIDTH + i * (env.COIN_RADIUS * 5), y + offset))

    # --- COLLISIONS & DRAWING ---
    def check_adversarial_collision(self, agent, agent_mask):
        agent_rect = pygame.Rect(
            agent.circle_pos[0] - agent.agent_radius, agent.circle_pos[1] - agent.agent_radius,
            agent.agent_radius * 2, agent.agent_radius * 2
        )

        for obs in self.obstacle_array:
            rotated_image, rect = obs.get_rect()
            if agent_rect.colliderect(rect):
                obstacle_mask = pygame.mask.from_surface(rotated_image)
                offset_x = int(rect.left - agent_rect.left)
                offset_y = int(rect.top - agent_rect.top)
                if obstacle_mask.overlap(agent_mask, (-offset_x, -offset_y)): return True

        for mis in self.missile_array:
            if mis.warning_timer <= 0 and agent_rect.colliderect(mis.get_rect()):
                return True
        return False

    def check_coin_collision(self, agent, agent_mask):
        for coin in self.coin_array[:]:
            distance = math.hypot(agent.circle_pos[0] - coin.x, agent.circle_pos[1] - coin.y)
            if distance < (agent.agent_radius + coin.radius):
                self.coin_array.remove(coin)
                return True
        return False

    def draw_threats(self, screen):
        for obs in self.obstacle_array: obs.draw(screen)
        for mis in self.missile_array: mis.draw(screen)

    def draw_coins(self, screen):
        for coin in self.coin_array: coin.draw(screen)