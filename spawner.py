
import random
import pygame
from coins import Coin
from obstacle import Obstacle
import environment as env

class Spawner:
    def __init__(self):
        self.obstacle_array = []
        self.coin_array = []
        self.sleep_time = random.randint(env.OBSTACLE_SPAWN_MIN, env.OBSTACLE_SPAWN_MAX)

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

    def update_obstacles(self):
        if self.sleep_time <= 0:
            self.spawn_obstacle()
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
        self.add_coin(Coin(env.SCREEN_WIDTH + env.COIN_RADIUS, random.randint(100, env.SCREEN_HEIGHT - env.PLATFORM_HEIGHT)))

    def update_coins(self):
        while len(self.coin_array) > 0 and self.coin_array[0].x + self.coin_array[0].radius < 0:
            self.coin_array.pop(0)
        if random.randint(1, 100) <= 1:
            self.spawn_coin()
        for coin in self.coin_array:
            coin.move()

    def draw_coins(self, screen):
        for coin in self.coin_array:
            coin.draw(screen)

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

