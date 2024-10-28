import pygame
import sys
from alaynaEnv import PongEnv

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 600
screen_height = 600
grid_size = 10
cell_size = screen_width // grid_size

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pong with Grid and Wall")

# Ball properties
ball_radius = cell_size // 4
ball_x, ball_y = screen_width // 2, screen_height // 2
ball_dx, ball_dy = 4, 4  # Ball speed

# Initialize the Pong environment
env = PongEnv(grid_size=grid_size)
env.reset()

# Function to draw grid
def draw_grid():
    for row in range(grid_size):
        for col in range(grid_size):
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, white, rect, 1)

# Function to draw wall
def draw_wall():
    wall_width = cell_size // 4  # Width of the wall markings
    for i in range(grid_size):
        pygame.draw.rect(screen, red, (0, i * cell_size, wall_width, cell_size))

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get the current state from the environment
    ball_x, ball_y, paddle_y, ball_dx, ball_dy = env.get_state()
    
    # Predict where the ball will hit on the right wall
    steps_to_wall = (grid_size - 1 - ball_x)
    predicted_ball_y = ball_y - (ball_dy * steps_to_wall)

    # Handle out-of-bounds for the predicted position (reflect back)
    if predicted_ball_y < 0:
        predicted_ball_y = -predicted_ball_y 
    elif predicted_ball_y >= grid_size:
        predicted_ball_y = (grid_size - 1) - (predicted_ball_y - (grid_size - 1)) 

    # Determine paddle action based on predicted ball position
    if predicted_ball_y < paddle_y:
        action = 1  # Move up
    elif predicted_ball_y > paddle_y:
        action = 2  # Move down
    else:
        action = 0  # Stay still

    # Execute the action in the environment
    new_state, reward, done = env.execute_action(action)

    # Clear the screen
    screen.fill(black)

    # Draw the grid, wall, and ball
    draw_grid()
    draw_wall()
    pygame.draw.circle(screen, white, (ball_x * cell_size + cell_size // 2, ball_y * cell_size + cell_size // 2), ball_radius)

    paddle_rect = pygame.Rect(screen_width - (cell_size // 2), paddle_y * cell_size, cell_size // 2, cell_size)
    pygame.draw.rect(screen, white, paddle_rect)

    # Update the display
    pygame.display.flip()

    # Set frame rate
    pygame.time.Clock().tick(10)
