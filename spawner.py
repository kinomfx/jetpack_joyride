
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
        # spawn timing for next unit (obstacle or pattern)
        self.sleep_time = random.randint(env.OBSTACLE_SPAWN_MIN, env.OBSTACLE_SPAWN_MAX)
        # Zone system: 1 = obstacle-only, 2 = pattern zone (coin-only or mixed)
        self.zone = 1
        self.zone_spawn_count = 0
        self.zone_threshold = random.randint(4, 8)  # switch after 4-8 spawns
        # pattern pool: methods that spawn patterns of coins/combined
        self.patterns = [
            self.pattern_coin_line,
            self.pattern_coin_arc,
            self.pattern_mixed_obstacle_coins,
            self.pattern_line_high,
            self.pattern_line_low,
            self.pattern_staggered_columns,
            self.pattern_zigzag,
            self.pattern_v_shape,
            self.pattern_double_arc,
            self.pattern_diagonal_steps,
            self.pattern_cross,
            self.pattern_snake,
            self.pattern_gap_pairs,
            self.pattern_obstacle_coins_top,
            self.pattern_obstacle_coins_bottom,
            self.pattern_rotating_center_combo,
            self.pattern_s_shape,
        ]

    def get_obstacles(self):
        return list(self.obstacle_array)

    def get_coins(self):
        return list(self.coin_array)

    # --- Obstacle methods ---
    def add_obstacle(self, obstacle):
        self.obstacle_array.append(obstacle)

    def remove_obstacle(self, obstacle):
        if obstacle in self.obstacle_array:
            self.obstacle_array.remove(obstacle)

    def spawn_obstacle(self):
        rotation = random.choice(env.OBSTACLE_ROTATIONS)
        height = random.randint(200, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT)
        # 30% chance to spawn a rotating obstacle
        if random.random() < 0.3:
            rotating = True
            rotation_speed = random.choice([-5, -3, -2, 2, 3, 5])
        else:
            rotating = False
            rotation_speed = 0
        self.add_obstacle(Obstacle(env.SCREEN_WIDTH + env.OBSTACLE_WIDTH, height, rotation, rotating, rotation_speed))
        self.sleep_time = random.randint(env.OBSTACLE_SPAWN_MIN, env.OBSTACLE_SPAWN_MAX)
        # zone accounting
        self.zone_spawn_count += 1
        self._maybe_switch_zone()

    def update_obstacles(self):
        # When it's time, spawn either an obstacle (zone 1) or a pattern (zone 2)
        if self.sleep_time <= 0:
            if self.zone == 1:
                self.spawn_obstacle()
            else:
                self.spawn_pattern()
        while len(self.obstacle_array) > 0 and self.obstacle_array[0].x + self.obstacle_array[0].height < 0:
            self.obstacle_array.pop(0)
        for obs in self.obstacle_array:
            obs.move()
        self.sleep_time -= 1

    def draw_obstacles(self, screen):
        for obs in self.obstacle_array:
            obs.draw(screen)

    def check_obstacle_collision(self, agent, agent_mask):
        for obs in self.obstacle_array:
            rotated_image, rect = obs.get_rect()
            obstacle_mask = pygame.mask.from_surface(rotated_image)
            offset_x = int(rect.left - (agent.circle_pos[0] - agent.agent_radius))
            offset_y = int(rect.top - (agent.circle_pos[1] - agent.agent_radius))
            if obstacle_mask.overlap(agent_mask, (-offset_x, -offset_y)):
                return True
        return False

    # --- Coin methods ---
    def add_coin(self, coin):
        self.coin_array.append(coin)

    def remove_coin(self, coin):
        if coin in self.coin_array:
            self.coin_array.remove(coin)

    def spawn_coin(self):
        # single coin spawn - still available for patterns to call
        self.add_coin(Coin(env.SCREEN_WIDTH + env.COIN_RADIUS, random.randint(100, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT)))

    def update_coins(self):
        while len(self.coin_array) > 0 and self.coin_array[0].x + self.coin_array[0].radius < 0:
            self.coin_array.pop(0)
        for coin in self.coin_array:
            coin.move()

    def draw_coins(self, screen):
        for coin in self.coin_array:
            coin.draw(screen)

    # --- Pattern spawning for zone 2 ---
    def spawn_pattern(self):
        # choose a pattern from the pool and execute it
        pattern = random.choice(self.patterns)
        # Debug: print which pattern is spawning
        try:
            print(f"[Spawner] Spawn pattern: {pattern.__name__}")
        except Exception:
            print("[Spawner] Spawn pattern: <unknown>")
        pattern()
        # give some time until next pattern or obstacle
        self.sleep_time = random.randint(env.OBSTACLE_SPAWN_MIN, env.OBSTACLE_SPAWN_MAX)
        self.zone_spawn_count += 1
        self._maybe_switch_zone()

    def pattern_coin_line(self, count=None):
        # horizontal line of coins (spaced in x)
        if count is None:
            count = random.randint(4, 7)
        y = random.randint(150, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 100)
        spacing = env.COIN_RADIUS * 4 + 10
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            self.add_coin(Coin(x, y))

    def pattern_coin_arc(self, count=None):
        # arc of coins using a sine-like offset
        if count is None:
            count = random.randint(5, 8)
        base_y = random.randint(200, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 150)
        spacing = env.COIN_RADIUS * 4 + 8
        amplitude = 60
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            # simple arc: move up then down
            t = i / max(1, count - 1)
            offset = int(amplitude * (1 - (2 * t - 1) ** 2))
            y = base_y - offset
            self.add_coin(Coin(x, y))

    def pattern_mixed_obstacle_coins(self):
        # spawn an obstacle with coins positioned above it
        rotation = random.choice(env.OBSTACLE_ROTATIONS)
        height = random.randint(220, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 80)
        # no rotation for mixed pattern to keep coin placement simple
        obs = Obstacle(env.SCREEN_WIDTH + env.OBSTACLE_WIDTH, height, rotation, False, 0)
        self.add_obstacle(obs)
        # spawn a small line of coins above the obstacle
        count = random.randint(3, 5)
        spacing = env.COIN_RADIUS * 4 + 6
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing + 20
            y = height - env.OBSTACLE_HEIGHT // 2 - 40
            self.add_coin(Coin(x, y))

    def _maybe_switch_zone(self):
        if self.zone_spawn_count >= self.zone_threshold:
            # toggle zone
            self.zone = 2 if self.zone == 1 else 1
            self.zone_spawn_count = 0
            self.zone_threshold = random.randint(4, 8)

    def pattern_line_high(self):
        # horizontal line near the top
        count = random.randint(5, 8)
        y = random.randint(80, 160)
        spacing = env.COIN_RADIUS * 4 + 6
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            self.add_coin(Coin(x, y))

    def pattern_line_low(self):
        # horizontal line lower but above platform
        count = random.randint(5, 8)
        y = random.randint(env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 160, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 80)
        spacing = env.COIN_RADIUS * 4 + 6
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            self.add_coin(Coin(x, y))

    def pattern_staggered_columns(self):
        # columns of coins staggered vertically
        cols = random.randint(3, 5)
        rows = random.randint(3, 5)
        x_spacing = env.COIN_RADIUS * 6 + 10
        y_spacing = env.COIN_RADIUS * 3 + 12
        start_x = env.SCREEN_WIDTH
        for c in range(cols):
            base_y = random.randint(120, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - (rows * y_spacing) - 20)
            for r in range(rows):
                x = start_x + c * x_spacing
                y = base_y + r * y_spacing + (c % 2) * (y_spacing // 2)
                self.add_coin(Coin(x, y))

    def pattern_zigzag(self):
        # zigzag pattern across the screen
        count = random.randint(6, 10)
        base_y = random.randint(140, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 140)
        spacing = env.COIN_RADIUS * 4 + 8
        amplitude = 80
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            y = base_y + (amplitude if i % 2 == 0 else -amplitude)
            self.add_coin(Coin(x, y))

    def pattern_v_shape(self):
        # V shape opening to the right
        arm = random.randint(4, 7)
        spacing = env.COIN_RADIUS * 4 + 8
        apex_y = random.randint(200, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 120)
        for i in range(arm):
            x1 = env.SCREEN_WIDTH + i * spacing
            y1 = apex_y - i * (env.COIN_RADIUS + 6)
            x2 = env.SCREEN_WIDTH + i * spacing
            y2 = apex_y + i * (env.COIN_RADIUS + 6)
            self.add_coin(Coin(x1, y1))
            self.add_coin(Coin(x2, y2))

    def pattern_double_arc(self):
        # two arcs one above the other
        count = random.randint(5, 8)
        spacing = env.COIN_RADIUS * 4 + 8
        top_base = random.randint(150, 260)
        bottom_base = top_base + 90
        amplitude = 50
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            t = i / max(1, count - 1)
            offset = int(amplitude * (1 - (2 * t - 1) ** 2))
            self.add_coin(Coin(x, top_base - offset))
            self.add_coin(Coin(x, bottom_base + offset))

    def pattern_diagonal_steps(self):
        # diagonal ascending or descending steps across the screen
        count = random.randint(6, 10)
        spacing = env.COIN_RADIUS * 4 + 8
        start_y = random.randint(140, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 140)
        step = random.randint(20, 60)
        direction = random.choice([1, -1])
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            y = start_y + direction * i * step
            # clamp y within safe bounds above platform
            y = max(80, min(y, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 80))
            self.add_coin(Coin(x, y))

    def pattern_cross(self):
        # cross shape (plus)
        center_y = random.randint(180, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 120)
        center_x = env.SCREEN_WIDTH + 120
        arm = 3
        spacing = env.COIN_RADIUS * 4 + 8
        # center
        self.add_coin(Coin(center_x, center_y))
        for i in range(1, arm + 1):
            self.add_coin(Coin(center_x + i * spacing, center_y))
            self.add_coin(Coin(center_x - i * spacing, center_y))
            self.add_coin(Coin(center_x, center_y + i * spacing))
            self.add_coin(Coin(center_x, center_y - i * spacing))

    def pattern_snake(self):
        # snake-like serpentine
        length = random.randint(8, 12)
        spacing = env.COIN_RADIUS * 4 + 6
        y = random.randint(140, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 140)
        dir_up = True
        amplitude = 60
        for i in range(length):
            x = env.SCREEN_WIDTH + i * spacing
            y_offset = amplitude if dir_up else -amplitude
            self.add_coin(Coin(x, y + y_offset))
            dir_up = not dir_up

    def pattern_gap_pairs(self):
        # pairs of coins with gaps to force weaving
        pairs = random.randint(4, 7)
        spacing_x = env.COIN_RADIUS * 6 + 12
        pair_gap = env.COIN_RADIUS * 6 + 24
        base_y_top = random.randint(120, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 180)
        base_y_bottom = base_y_top + random.randint(60, 120)
        start_x = env.SCREEN_WIDTH
        for i in range(pairs):
            x_left = start_x + i * spacing_x * 2
            # left pair (top and bottom)
            self.add_coin(Coin(x_left, base_y_top))
            self.add_coin(Coin(x_left, base_y_bottom))
            # right pair offset by pair_gap
            x_right = x_left + pair_gap
            self.add_coin(Coin(x_right, base_y_top))
            self.add_coin(Coin(x_right, base_y_bottom))

    def pattern_obstacle_coins_top(self):
        # horizontal obstacle with a bunch of coins above it
        rotation = 0
        # place obstacle around mid-height but leave room above for coins
        height = random.randint(260, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 140)
        obs_x = env.SCREEN_WIDTH + env.OBSTACLE_WIDTH
        obs = Obstacle(obs_x, height, rotation, False, 0)
        self.add_obstacle(obs)
        # place coins above obstacle, spaced horizontally
        count = random.randint(4, 7)
        spacing = env.COIN_RADIUS * 4 + 8
        top_y = height - obs.height // 2 - env.COIN_RADIUS - 20
        start_x = env.SCREEN_WIDTH
        for i in range(count):
            x = start_x + i * spacing
            y = max(80, top_y - random.randint(0, 20))
            self.add_coin(Coin(x, y))

    def pattern_obstacle_coins_bottom(self):
        # horizontal obstacle with a bunch of coins below it
        rotation = 0
        height = random.randint(180, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 200)
        obs_x = env.SCREEN_WIDTH + env.OBSTACLE_WIDTH
        obs = Obstacle(obs_x, height, rotation, False, 0)
        self.add_obstacle(obs)
        # place coins below obstacle, spaced horizontally
        count = random.randint(4, 7)
        spacing = env.COIN_RADIUS * 4 + 8
        bottom_y = height + obs.height // 2 + env.COIN_RADIUS + 20
        start_x = env.SCREEN_WIDTH
        for i in range(count):
            x = start_x + i * spacing
            y = min(env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 40, bottom_y + random.randint(0, 20))
            self.add_coin(Coin(x, y))

    def pattern_rotating_center_combo(self):
        # rotating obstacle in middle with coins above and below (safe margin)
        rotation = random.randint(0, 360)
        height = random.randint(220, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 160)
        obs_x = env.SCREEN_WIDTH + env.OBSTACLE_WIDTH
        rotation_speed = random.choice([-4, -3, 3, 4])
        obs = Obstacle(obs_x, height, rotation, True, rotation_speed)
        self.add_obstacle(obs)
        # safe margin: place coins beyond obstacle half-height + extra margin
        margin = 18
        top_y = height - obs.height // 2 - env.COIN_RADIUS - margin
        bottom_y = height + obs.height // 2 + env.COIN_RADIUS + margin
        count = random.randint(3, 6)
        spacing = env.COIN_RADIUS * 4 + 8
        start_x = env.SCREEN_WIDTH
        for i in range(count):
            x = start_x + i * spacing
            self.add_coin(Coin(x, max(80, top_y - random.randint(0, 16))))
            self.add_coin(Coin(x, min(env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 40, bottom_y + random.randint(0, 16))))

    def pattern_s_shape(self):
        # S-shaped pattern using sine wave across several points
        count = random.randint(8, 12)
        spacing = env.COIN_RADIUS * 4 + 8
        amplitude = random.randint(60, 100)
        base_y = random.randint(160, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 160)
        for i in range(count):
            x = env.SCREEN_WIDTH + i * spacing
            t = i / max(1, count - 1)
            # sine-based S shape (flip sign halfway)
            y = base_y + int(amplitude * math.sin(2 * math.pi * t * 1.0))
            # clamp
            y = max(80, min(y, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT - 80))
            self.add_coin(Coin(x, y))

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
                self.remove_coin(coin)
                return True
        return False

