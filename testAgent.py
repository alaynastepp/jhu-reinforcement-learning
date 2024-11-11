import numpy as np
""""
Test agent for testing visualization class - knows the whole environment so will always no where to place paddle
"""
class TestAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, state_index):
        # Get the current environment state (ball and paddle positions, velocities)
        ball_x, ball_y, paddle_y, ball_dx, ball_dy = self.env.get_state()

        # Predict steps to the right wall
        steps_to_wall = (self.env.grid_size - 1 - ball_x)
        # Calculate predicted y-position of the ball at the right wall
        predicted_ball_y = ball_y - (ball_dy * steps_to_wall)

        # Handle reflection if predicted position goes out-of-bounds
        if predicted_ball_y < 0:
            predicted_ball_y = -predicted_ball_y
        elif predicted_ball_y >= self.env.grid_size:
            predicted_ball_y = (self.env.grid_size - 1) - (predicted_ball_y - (self.env.grid_size - 1))

        # Determine the action to align paddle with predicted ball position
        if predicted_ball_y < paddle_y:
            return 1  # Move paddle up
        elif predicted_ball_y > paddle_y:
            return 2  # Move paddle down
        else:
            return 0  # Stay still

    def update(self, next_state_index, reward):
        # Placeholder for agents that learn from experience (e.g., Q-learning)
        # TestAgent does not learn, so no updates are made here.
        pass
