import numpy as np

class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None, max_steps=100):
        self.grid_size = grid_size
        self.initial_ball_dx = ball_dx
        self.initial_ball_dy = ball_dy
        self.initial_ball_x = ball_x if ball_x is not None else self.grid_size // 2 # if not specificed, ball starts in the center
        self.initial_ball_y = ball_y if ball_y is not None else self.grid_size // 2 # if not specificed, ball starts in the center
        self.current_step = 0
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """"
        Reset grid to the inital state.
        
        :return state (list): Current state: ball position, paddle position, and velocity
        """
        # Reset ball position and direction to the initial values
        self.ball_x = self.initial_ball_x
        self.ball_y = self.initial_ball_y
        self.ball_dx = self.initial_ball_dx
        self.ball_dy = self.initial_ball_dy
        
        # Reset the paddle position
        self.paddle_y = self.grid_size // 2
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Get the current state (ball position, paddle position, and velocity)
        
        :return state (list): Current state: ball position, paddle position, and velocity
        """
        return (self.ball_x, self.ball_y, self.paddle_y, self.ball_dx, self.ball_dy)

    def get_number_of_states(self):
        """
        Number of possible states (based on ball position, paddle position, and ball velocity)
        
        :return (int): Total number of states
        """
        return (self.grid_size ** 2) * self.grid_size * 9  # ball positions * paddle positions * velocities

    def get_number_of_actions(self):
        """ 
        3 possible actions: move up, move down, or stay still
        """
        return 3

    def execute_action(self, action):
        """
        Executes the action and returns the reward.
        Actions: 0 = stay still, 1 = move up, 2 = move down
        
        :param action (int): The current action of the agent.
        :return state (list): Current state: ball position, paddle position, and velocity
        :return reward (int): The action selected
        :return done (bool): The action selected
        """
        if action == 1 and self.paddle_y > 0:
            self.paddle_y -= 1
        elif action == 2 and self.paddle_y < self.grid_size - 1:
            self.paddle_y += 1
        
        # move the ball
        self.ball_x += self.ball_dx #add dx for movement to the right
        self.ball_y -= self.ball_dy #subtract dy for upward movement
        
        # ball bounces off the left wall (|) on the far-left side
        if self.ball_x == 0:
            self.ball_dx *= -1

        # ball bounces off the walls (top and bottom)
        if self.ball_y == 0 or self.ball_y == self.grid_size - 1:
            self.ball_dy *= -1

        # ball reaches the right side 
        if self.ball_x == self.grid_size - 1:
            if self.paddle_y == self.ball_y:
                self.score += 1
                reward = +1
                self.ball_dx *= -1  # reverse direction
                
                # Handle ball angle change based on paddle movement and ball direction
                if self.ball_dy < 0:  # Ball coming down
                    if action == 1:  # Paddle moving up
                        self.ball_dy = 0  # Normal bounce: send ball back in opposite angle
                    elif action == 2:  # Paddle moving down
                        self.ball_dy = -1  # Send ball back up at same 45-degree angle
                    elif action == 0:  # Paddle stationary
                        self.ball_dy = -1  # Send ball back up at same 45-degree angle

                elif self.ball_dy > 0:  # Ball coming up
                    if action == 1:  # Paddle moving up
                        self.ball_dy = 1  # Send ball back down at same 45-degree angle
                    elif action == 2:  # Paddle moving down
                        self.ball_dy = 0  # Send ball back in a straight vertical line
                    elif action == 0:  # Paddle stationary
                        self.ball_dy = 1  # Send ball back down at same 45-degree angle

                elif self.ball_dy == 0:  # Ball coming straight
                    if action == 1:  # Paddle moving up
                        self.ball_dy = -1  # Send ball back at a 45-degree angle up
                    elif action == 2:  # Paddle moving down
                        self.ball_dy = 1  # Send ball back at a 45-degree angle down
                    elif action == 0:  # Paddle stationary
                        self.ball_dy = 0  # Send ball back in a straight vertical line
            else:
                # paddle missed the ball
                self.done = True
                reward = -1
        else:
            reward = 0  # no point scored, continue game

        # make sure ball stays within grid bounds
        self.ball_x = max(0, min(self.ball_x, self.grid_size - 1))
        self.ball_y = max(0, min(self.ball_y, self.grid_size - 1))
        
        self.current_step += 1
        # Check if the game has timed out
        if self.current_step >= self.max_steps:
            self.done = True  
            reward = 0  

        return self.get_state(), reward, self.done
    
    def render(self): 
        """
        Visualize the grid with ball and paddle
        """
        # grid where all ball positions are '0'
        grid = np.full((self.grid_size, self.grid_size), "0", dtype=str)

        # add the left-hand wall 
        grid[:, 0] = "|"

        # add the right-hand side where the paddle is
        grid[:, -1] = "x" 

        # add the paddle 
        grid[self.paddle_y, -1] = "p"
        
        # add the ball 
        grid[self.ball_y, self.ball_x] = "*"

        for row in grid:
            print(" ".join(row))
        print("\n")



# semi-smart algorithm to test if bounce angle logics works
if __name__ == "__main__":
    env = PongEnv()
    env.reset()
    print("Starting grid:")
    env.render()

    steps = 50
    for step in range(steps):
        ball_x, ball_y, paddle_y, ball_dx, ball_dy = env.get_state()

        # Predict how many steps the ball will travel until it hits the right wall
        steps_to_wall = (env.grid_size - 1 - ball_x)
        # Calculate the predicted ball position when it hits the wall
        predicted_ball_y = ball_y - (ball_dy * steps_to_wall)

        # Handle out-of-bounds for the predicted position (reflect back)
        if predicted_ball_y < 0:
            predicted_ball_y = -predicted_ball_y 
        elif predicted_ball_y >= env.grid_size:
            predicted_ball_y = (env.grid_size - 1) - (predicted_ball_y - (env.grid_size - 1)) 

        # Move the paddle to the predicted position, ensuring it stays within bounds
        if predicted_ball_y < paddle_y:
            action = 1  # Move up
        elif predicted_ball_y > paddle_y:
            action = 2  # Move down
        else:
            action = 0  # Stay still

        # Execute the action and get the new state and reward
        new_state, reward, done = env.execute_action(action)

        # Print the step information
        print(f"Step: {step + 1}, State: {new_state}, Reward: {reward}, Done: {done}")

        # Render the current state of the game
        env.render()

        # Check if the game is over
        if done:
            print("Game Over!")
            break

    # Check the final score to determine win/loss
    if env.score > 0:
        print("You won! Final score:", env.score)
    else:
        print("You lost! Final score:", env.score)