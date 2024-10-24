import pygame
from alaynaEnv import PongEnv 

# Constants for visualization
WINDOW_SIZE = 500
GRID_SIZE = 10
CELL_SIZE = WINDOW_SIZE // GRID_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

def draw_grid(screen):
    """ Draw the grid background """
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (0, y), (WINDOW_SIZE, y))

def draw_paddle(screen, paddle_y):
    """ Draw the paddle """
    paddle_rect = pygame.Rect(WINDOW_SIZE - CELL_SIZE, paddle_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, paddle_rect)

def draw_ball(screen, ball_x, ball_y):
    """ Draw the ball """
    ball_rect = pygame.Rect(ball_x * CELL_SIZE, ball_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, ball_rect)

def visualize_pong():
    pygame.init()
    
    # Set up display
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Pong Visualization")

    # Initialize the Pong Environment
    env = PongEnv(grid_size=GRID_SIZE)
    env.reset()

    clock = pygame.time.Clock()
    running = True
    steps = 50
    action = 0  # Default action (stay still)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(BLACK)
        draw_grid(screen)

        # Get the current state
        ball_x, ball_y, paddle_y, ball_dx, ball_dy = env.get_state()

        # Visualize the paddle and ball
        draw_paddle(screen, paddle_y)
        draw_ball(screen, ball_x, ball_y)
        
        # Predict how many steps the ball will travel until it hits the right wall
        steps_to_wall = (env.grid_size - 1 - ball_x)
        predicted_ball_y = ball_y - (ball_dy * steps_to_wall)

        # Handle out-of-bounds for the predicted position (reflect back)
        if predicted_ball_y < 0:
            predicted_ball_y = -predicted_ball_y
        elif predicted_ball_y >= env.grid_size:
            predicted_ball_y = (env.grid_size - 1) - (predicted_ball_y - (env.grid_size - 1))

        # Move the paddle to the predicted position
        if predicted_ball_y < paddle_y:
            action = 1  # Move up
        elif predicted_ball_y > paddle_y:
            action = 2  # Move down
        else:
            action = 0  # Stay still

        # Execute the action and update state
        new_state, reward, done = env.execute_action(action)

        # Update display
        pygame.display.flip()

        # Check if game is over
        if done:
            print("Game Over!")
            running = False

        # Limit frames per second
        clock.tick(5)  # Adjust FPS as needed

    pygame.quit()

if __name__ == "__main__":
    visualize_pong()
