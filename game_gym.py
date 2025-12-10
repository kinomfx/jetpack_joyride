"""
game_gym.py - Gym-like wrapper for the JetPack JoyRide game

This wrapper provides a standard interface for RL training:
- reset(): Start a new episode
- step(action): Execute action and return (state, reward, done, info)
- render(): Display the game
- close(): Clean up resources
"""

import pygame
import numpy as np
from game import Game, GameState
from state_extractor import extract_state_vector


class GameGym:
    """
    OpenAI Gym-like environment wrapper for JetPack JoyRide

    This allows the DQN agent to interact with the game using a standard interface.
    """

    def __init__(self):
        """Initialize the game environment"""
        pygame.init()
        self.game = None
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Track previous state for reward calculation
        self.prev_score = 0
        self.prev_coins = 0

    def reset(self):
        """
        Reset the environment to start a new episode

        Returns:
            state (np.array): Initial state vector
        """
        # Clear any existing game
        if self.game is not None:
            pygame.event.clear()

        # Create new game instance
        self.game = Game(start_state=GameState.RUNNING)

        # Reset tracking variables
        self.prev_score = 0
        self.prev_coins = 0

        # Extract and return initial state
        state = extract_state_vector(self.game)
        return state

    def step(self, action):
        """
        Execute one time step in the environment

        Args:
            action (int): 0 = do nothing, 1 = press space (fly up)

        Returns:
            state (np.array): Next state vector
            reward (float): Reward for this step
            done (bool): Whether episode is finished
            info (dict): Additional information
        """
        # Handle pygame events to prevent window freezing
        pygame.event.pump()

        # Execute action
        if action == 1:
            self.game.agent.move_up(self.game.upward_force)

        # Update game state
        self.game.apply_gravity()

        # Update speed multiplier based on score
        mult = 1.0 + (self.game.score / self.game.spawner.speed_multiplier)
        mult = min(mult, 2.25)  # MAX_SPEED_MULTIPLIER
        self.game.spawner.speed_multiplier = mult

        # Get agent mask for collision detection
        agent_mask = self.game.agent.get_mask()

        # Update obstacles and check collision
        self.game.spawner.update_obstacles()
        collision = self.game.spawner.check_obstacle_collision(self.game.agent, agent_mask)

        # Update coins and check collection
        self.game.spawner.update_coins()
        coin_collected = self.game.spawner.check_coin_collision(self.game.agent, agent_mask)
        if coin_collected:
            self.game.coin_count += 1

        # Increment score
        self.game.score += 1

        # Update agent vertical velocity estimate
        prev_y = self.game.agent.circle_pos[1]
        self.game._agent_vy = 0.0  # Will be updated in next frame

        # Check if episode is done
        done = collision or self.game.state != GameState.RUNNING

        # Get next state
        next_state = extract_state_vector(self.game)

        # Calculate reward (simple version - can be customized in training loop)
        reward = 0.1  # Survival bonus
        if collision:
            reward = -100.0
        elif coin_collected:
            reward += 10.0

        # Additional info
        info = {
            'score': self.game.score,
            'coins': self.game.coin_count,
            'collision': collision
        }

        return next_state, reward, done, info

    def render(self):
        """Render the game screen"""
        if self.game is not None:
            self.game.render()
            self.clock.tick(self.fps)

    def close(self):
        """Clean up resources"""
        pygame.quit()

    def set_fps(self, fps):
        """Set the rendering frame rate"""
        self.fps = fps


# Test the wrapper
if __name__ == "__main__":
    """Test the GameGym wrapper with random actions"""
    env = GameGym()

    print("Testing GameGym wrapper with random actions...")
    print("Press Ctrl+C to stop")
    print("-" * 50)

    try:
        for episode in range(5):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            print(f"\nEpisode {episode + 1}")

            while not done:
                # Random action
                action = np.random.randint(0, 2)

                # Step
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                # Render
                env.render()

                # Handle quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        exit()

                state = next_state

            print(f"  Score: {info['score']}, Coins: {info['coins']}, Steps: {steps}, Total Reward: {total_reward:.1f}")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    env.close()
    print("\nTest completed!")