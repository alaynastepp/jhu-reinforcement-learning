import pygame
import sys

class PongVisualizer:
    def __init__(self, grid_size=10, cell_size=60):
        pygame.init()
        self.screen_width = grid_size * cell_size
        self.screen_height = grid_size * cell_size
        self.grid_size = grid_size
        self.cell_size = cell_size

        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)

        # Ball properties
        self.ball_radius = cell_size // 4

        # Set up the display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pong with Grid and Wall")

    def draw_grid(self):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.white, rect, 1)

    def draw_wall(self):
        wall_width = self.cell_size // 4
        for i in range(self.grid_size):
            pygame.draw.rect(self.screen, self.red, (0, i * self.cell_size, wall_width, self.cell_size))

    def render(self, ball_position, paddle_position):
        # Handle Pygame events for quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Clear the screen
        self.screen.fill(self.black)

        # Draw the grid and wall
        self.draw_grid()
        self.draw_wall()

        # Draw the ball
        ball_x, ball_y = ball_position
        pygame.draw.circle(self.screen, self.white, (ball_x * self.cell_size + self.cell_size // 2, ball_y * self.cell_size + self.cell_size // 2), self.ball_radius)

        # Draw the paddle
        paddle_rect = pygame.Rect(self.screen_width - (self.cell_size // 2), paddle_position * self.cell_size, self.cell_size // 2, self.cell_size)
        pygame.draw.rect(self.screen, self.white, paddle_rect)

        # Update the display
        pygame.display.flip()
        
        # Set frame rate
        pygame.time.Clock().tick(5)

    def close(self):
        pygame.quit()
        