import numpy as np

class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None, max_steps=10000):
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
        self.current_step = 0

        return self.get_state()
    
    def get_state_index(self):
        """
        Convert the current state (ball position, paddle position, and velocity) into a unique index.

        :return (int): The unique index representing the current state.
        """

        # explanation:
        # This part encodes the ball’s position on a 2D grid (like coordinates). Suppose the grid_size is 10.
        # If ball_x = 3 and ball_y = 4, then ball_pos_index = 3 * 10 + 4 = 34.
        # This turns a (3, 4) coordinate into a unique number, 34, which represents a linearized position in a 1D array for easier indexing.
        ball_pos = self.ball_x * self.grid_size + self.ball_y
        
        # This represents the paddle's vertical position. Since it can only move vertically, self.paddle_y is enough to capture its state.
        paddle_pos = self.paddle_y
        
        # This converts the ball’s direction (dx and dy) into a single index.
        # Here, self.ball_dx and self.ball_dy might be values like -1, 0, or 1 (representing left, neutral, or right for dx and up, neutral, or down for dy).
        # By shifting both dx and dy by 1 (to 0, 1, 2), and then using 3 * dx + dy, this creates a unique value for each possible velocity combination:
        # For example, if dx = -1 and dy = 1, then ball_velocity_index = (0) * 3 + 2 = 2.
        # This approach creates a unique index between 0 and 8 for the 3x3 grid of (dx, dy) values.
        ball_velocity = (self.ball_dx + 1) * 3 + (self.ball_dy + 1)  # Encode dx and dy values

        # Combine them all
        # Multiplying each part by different factors ensures that each combination of 
        # ball position, paddle position, and ball velocity has a unique state_index.
        return ball_pos * 90 + paddle_pos * 9 + ball_velocity

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
        # max ball position is 9,9 
        max_ball_pos = (self.grid_size - 1) * 10 + (self.grid_size - 1)

        # max paddle position is 9
        max_paddle = self.grid_size - 1

        # with explanation above in get_state_index
        # ball_dx = 1, ball_dy = 1
        max_ball_velocity = (1 + 1) * 3 + (1 + 1)
        
        return max_ball_pos * 90 + max_paddle * 9 + max_ball_velocity

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
                reward = +5
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
                reward = -25
        else:
            reward = 0  # no point scored, continue game

        # make sure ball stays within grid bounds
        self.ball_x = max(0, min(self.ball_x, self.grid_size - 1))
        self.ball_y = max(0, min(self.ball_y, self.grid_size - 1))
        
        self.current_step += 1
        # Check if the game has timed out
        if self.current_step >= self.max_steps:
            self.done = True  
            reward = 100

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
    
    def get_terminal_states(self):
        """
        Gets all the possible terminal states that end the game

        :return (list): list of ints that represent terminal states
        """
        raise NotImplementedError
    
    def get_score(self):
        """
        Gets the current score

        :return (ing): current score
        """
        return self.score

if __name__ == '__main__':
    env = PongEnv(grid_size=10)
    state = env.reset()
    print("Initial state:", state)
    print("Total states: ", env.get_number_of_states())
    print("Total actions: ", env.get_number_of_actions())