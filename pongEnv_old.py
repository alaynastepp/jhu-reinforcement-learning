import numpy as np

class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None, max_steps=10000):
        self.grid_size = grid_size
        self.initial_ball_dx = ball_dx
        self.initial_ball_dy = ball_dy
        self.initial_ball_x = ball_x if ball_x is not None else self.grid_size // 2 # if not specificed, ball starts in the center
        self.initial_ball_y = ball_y if ball_y is not None else self.grid_size // 2 # if not specificed, ball starts in the center
        
        self.paddle_y = self.grid_size // 2
        
        self.score = 0
        self.done = False
        self.current_step = 0
        self.max_steps = max_steps

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
        return ball_pos * (self.grid_size * 9) + paddle_pos * 9 + ball_velocity
    
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
        grid_size = self.grid_size
        num_ball_positions = grid_size * grid_size  # ball_x and ball_y
        num_paddle_positions = grid_size  # paddle_y
        num_velocities = 3 * 3  # ball_dx and ball_dy (-1, 0, 1 for both)

        total_states = num_ball_positions * num_paddle_positions * num_velocities
        return total_states

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
        
        # add the ball 
        grid[self.ball_y, self.ball_x] = "*"
        
        # add the paddle 
        grid[self.paddle_y, -1] = "p"

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

def verify_unique_indices(env):
    unique_indices = set()
    duplicates = False
    grid_size = env.grid_size  

    # Iterate over all possible values for ball position, velocity, and paddle position
    for ball_x in range(grid_size):
        for ball_y in range(grid_size):
            for ball_dx in [-1, 0, 1]:   # Assuming velocities are only -1 or +1
                for ball_dy in [-1, 0, 1]:
                    for paddle_y in range(grid_size):
                        # Set the environment to this state
                        env.ball_x, env.ball_y = ball_x, ball_y
                        env.ball_dx, env.ball_dy = ball_dx, ball_dy
                        env.paddle_y = paddle_y

                        # Calculate the state index
                        state_index = env.get_state_index()

                        # Check for uniqueness of the state index
                        if state_index in unique_indices:
                            print(f"Duplicate index found: "
                                  f"Ball position ({ball_x}, {ball_y}), "
                                  f"Velocity ({ball_dx}, {ball_dy}), "
                                  f"Paddle position {paddle_y} -> State Index: {state_index}")
                            duplicates = True
                        else:
                            unique_indices.add(state_index)

                        total_states = env.get_number_of_states()
                        assert (0 <= state_index < total_states), f"State index {state_index} out of bounds (0 to {total_states - 1})"

    # Final summary
    if duplicates:
        print("There are duplicates in the state index calculations.")
    else:
        print("All state indices are unique. `get_state_index` logic appears correct.")
    print(f"Total unique states checked: {len(unique_indices)}")

if __name__ == '__main__':
    env = PongEnv(grid_size=10)
    state = env.reset()
    print("Initial state:", state)
    print("Total states: ", env.get_number_of_states())
    print("Total actions: ", env.get_number_of_actions())
    verify_unique_indices(env)