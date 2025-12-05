import pygame
from score_manager import ScoreManager

class HUD:
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.score_pos = (self.screen_width - 25, 10)
        self.coin_pos = (self.screen_width - 25, 40)
        self.score_manager = ScoreManager()

    def draw(self, score, coin_count):
        font = pygame.font.SysFont("comicsans", 30)
        text = font.render("Score: " + str(int(score)), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topright = self.score_pos
        self.screen.blit(text, text_rect)
        coin_text = font.render("Coins: " + str(int(coin_count)), True, (255, 255, 255))
        coin_rect = coin_text.get_rect()
        coin_rect.topright = self.coin_pos
        self.screen.blit(coin_text, coin_rect)

    def draw_endscreen(self):
        self.screen.fill("black")
        font = pygame.font.SysFont("comicsans", 40)
        text1 = font.render("Game Over!", True, (255, 0, 0))
        text2 = font.render("Press ESC to return to menu", True, (255, 255, 255))
        text1_rect = text1.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        text2_rect = text2.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text1, text1_rect)
        self.screen.blit(text2, text2_rect)
        pygame.display.update()

    def draw_startscreen(self):
        self.screen.fill("black")
        title_font = pygame.font.SysFont("comicsans", 60)
        menu_font = pygame.font.SysFont("comicsans", 40)
        
        title = title_font.render("JetPack JoyRide", True, (255, 223, 0))
        title_rect = title.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 150))
        self.screen.blit(title, title_rect)
        
        start_text = menu_font.render("Press SPACE to Start", True, (255, 255, 255))
        start_rect = start_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(start_text, start_rect)
        
        highscore_text = menu_font.render("Press H for High Scores", True, (255, 255, 255))
        highscore_rect = highscore_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
        self.screen.blit(highscore_text, highscore_rect)
        
        quit_text = menu_font.render("Press ESC to Quit", True, (255, 255, 255))
        quit_rect = quit_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 160))
        self.screen.blit(quit_text, quit_rect)
        
        pygame.display.update()

    def draw_highscores(self):
        self.screen.fill("black")
        title_font = pygame.font.SysFont("comicsans", 50)
        score_font = pygame.font.SysFont("comicsans", 35)
        small_font = pygame.font.SysFont("comicsans", 25)
        
        title = title_font.render("High Scores", True, (255, 223, 0))
        title_rect = title.get_rect(center=(self.screen_width // 2, 50))
        self.screen.blit(title, title_rect)
        
        scores = self.score_manager.get_high_scores()
        y_offset = 150
        
        if not scores:
            no_scores = score_font.render("No high scores yet!", True, (255, 255, 255))
            no_scores_rect = no_scores.get_rect(center=(self.screen_width // 2, y_offset))
            self.screen.blit(no_scores, no_scores_rect)
        else:
            for i, score_data in enumerate(scores, 1):
                score_text = score_font.render(f"{i}. Score: {score_data['score']} | Coins: {score_data['coins']}", True, (255, 255, 255))
                score_rect = score_text.get_rect(center=(self.screen_width // 2, y_offset))
                self.screen.blit(score_text, score_rect)
                y_offset += 60
        
        back_text = small_font.render("Press any key to go back", True, (200, 200, 200))
        back_rect = back_text.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
        self.screen.blit(back_text, back_rect)
        
        pygame.display.update()