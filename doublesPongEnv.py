import numpy as np

class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None, max_steps=10000):
        self.grid_size = grid_size
        self.initial_ball_dx = ball_dx
        self.initial_ball_dy = ball_dy
        self.initial_ball_x = ball_x if ball_x is not None else self.grid_size // 2
        self.initial_ball_y = ball_y if ball_y is not None else self.grid_size // 2
        
        self.paddle_y_right = self.grid_size // 2
        self.paddle_y_left = self.grid_size // 2
        
        self.score_left = 0
        self.score_right = 0
        self.done = False
        self.current_step = 0
        self.max_steps = max_steps

    def reset(self):
        """"
        Reset grid to the inital state.
        
        :return state (list): Current state: ball position, paddle position, and velocity
        """
        self.ball_x = self.initial_ball_x
        self.ball_y = self.initial_ball_y
        self.ball_dx = self.initial_ball_dx
        self.ball_dy = self.initial_ball_dy
        
        self.paddle_y_right = self.grid_size // 2
        self.paddle_y_left = self.grid_size // 2
        
        self.score_left = 0
        self.score_right = 0
        self.done = False
        self.current_step = 0

        return self.get_state()
        
    def get_number_of_states(self) -> int:
        """
        Calculates the total number of states possible in the Pong environment.
        """
        grid_size = self.grid_size
        num_ball_positions = grid_size * grid_size  # ball_x and ball_y
        num_velocities = 3 * 3  # ball_dx and ball_dy (-1, 0, 1 for both)
        num_paddle_positions = grid_size * grid_size  # paddle_y_left and paddle_y_right

        total_states = num_ball_positions * num_velocities * num_paddle_positions
        return total_states

    def get_state_index(self):
        ball_x = self.ball_x
        ball_y = self.ball_y
        paddle_y_left = self.paddle_y_left
        paddle_y_right = self.paddle_y_right
        ball_dx = self.ball_dx
        ball_dy = self.ball_dy

        # Map -1 -> 0, 0 -> 1, 1 -> 2 for dx and dy
        ball_dx_index = ball_dx + 1  # Maps -1 -> 0, 0 -> 1, 1 -> 2
        ball_dy_index = ball_dy + 1  # Maps -1 -> 0, 0 -> 1, 1 -> 2

        # Linear approach: state index calculation
        index = (ball_x
                + ball_y * self.grid_size
                + paddle_y_left * self.grid_size * self.grid_size
                + paddle_y_right * self.grid_size * self.grid_size * self.grid_size
                + ball_dx_index * self.grid_size * self.grid_size * self.grid_size * self.grid_size
                + ball_dy_index * self.grid_size * self.grid_size * self.grid_size * self.grid_size * 3)  # Adjust the scale here

        return index


    def get_state(self):
        """
        Get the current state (ball position, two paddle positions, and velocity)
        
        :return state (list): Current state: ball position, two paddle paddle positions, and velocity
        """
        return (self.ball_x, self.ball_y, self.paddle_y_left, self.paddle_y_right, self.ball_dx, self.ball_dy)

    def get_number_of_actions(self):
        """ 
        3 possible actions: move up, move down, or stay still
        """
        return 3  # Actions: 0 = stay, 1 = move up, 2 = move down

    def execute_action(self, action_left, action_right):
        """
        Executes the action and returns the reward.
        Actions: 0 = stay still, 1 = move up, 2 = move down
        
        :param action_left (int): The current action of the agent on the left side.
        :param action_right (int): The current action of the agent on the right side.
        :return state (list): Current state: ball position, paddle position, and velocity
        :return reward (int): The action selected
        :return done (bool): The action selected
        """
        if action_left == 1 and self.paddle_y_left > 0:
            self.paddle_y_left -= 1
        elif action_left == 2 and self.paddle_y_left < self.grid_size - 1:
            self.paddle_y_left += 1
        
        if action_right == 1 and self.paddle_y_right > 0:
            self.paddle_y_right -= 1
        elif action_right == 2 and self.paddle_y_right < self.grid_size - 1:
            self.paddle_y_right += 1
        
        # Move the ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball bounces off the top or bottom
        if self.ball_y == 0 or self.ball_y == self.grid_size - 1:
            self.ball_dy *= -1

        reward_left = 0
        reward_right = 0

        # Ball bounces off left paddle
        if self.ball_x == 0:
            if self.paddle_y_left == self.ball_y:
                self.score_left += 1
                reward_left = +5
                self.ball_dx *= -1
                # Handle ball angle change based on paddle movement and ball direction
                if self.ball_dy < 0:  # Ball coming down
                    if action_left == 1:  # Paddle moving up
                        self.ball_dy = 0  # Normal bounce: send ball back in opposite angle
                    elif action_left == 2:  # Paddle moving down
                        self.ball_dy = -1  # Send ball back up at same 45-degree angle
                    elif action_left == 0:  # Paddle stationary
                        self.ball_dy = -1  # Send ball back up at same 45-degree angle

                elif self.ball_dy > 0:  # Ball coming up
                    if action_left == 1:  # Paddle moving up
                        self.ball_dy = 1  # Send ball back down at same 45-degree angle
                    elif action_left == 2:  # Paddle moving down
                        self.ball_dy = 0  # Send ball back in a straight vertical line
                    elif action_left == 0:  # Paddle stationary
                        self.ball_dy = 1  # Send ball back down at same 45-degree angle

                elif self.ball_dy == 0:  # Ball coming straight
                    if action_left == 1:  # Paddle moving up
                        self.ball_dy = -1  # Send ball back at a 45-degree angle up
                    elif action_left == 2:  # Paddle moving down
                        self.ball_dy = 1  # Send ball back at a 45-degree angle down
                    elif action_left == 0:  # Paddle stationary
                        self.ball_dy = 0  # Send ball back in a straight vertical line
            else:
                reward_left = -25
                self.done = True

        # Ball bounces off right paddle
        elif self.ball_x == self.grid_size - 1:
            if self.paddle_y_right == self.ball_y:
                self.score_right += 1
                reward_right = +5
                self.ball_dx *= -1
                # Handle ball angle change based on paddle movement and ball direction
                if self.ball_dy < 0:  # Ball coming down
                    if action_right == 1:  # Paddle moving up
                        self.ball_dy = 0  # Normal bounce: send ball back in opposite angle
                    elif action_right == 2:  # Paddle moving down
                        self.ball_dy = -1  # Send ball back up at same 45-degree angle
                    elif action_right == 0:  # Paddle stationary
                        self.ball_dy = -1  # Send ball back up at same 45-degree angle

                elif self.ball_dy > 0:  # Ball coming up
                    if action_right == 1:  # Paddle moving up
                        self.ball_dy = 1  # Send ball back down at same 45-degree angle
                    elif action_right == 2:  # Paddle moving down
                        self.ball_dy = 0  # Send ball back in a straight vertical line
                    elif action_right == 0:  # Paddle stationary
                        self.ball_dy = 1  # Send ball back down at same 45-degree angle

                elif self.ball_dy == 0:  # Ball coming straight
                    if action_right == 1:  # Paddle moving up
                        self.ball_dy = -1  # Send ball back at a 45-degree angle up
                    elif action_right == 2:  # Paddle moving down
                        self.ball_dy = 1  # Send ball back at a 45-degree angle down
                    elif action_right == 0:  # Paddle stationary
                        self.ball_dy = 0  # Send ball back in a straight vertical line
            else:
                reward_right = -25
                self.done = True

        # make sure ball stays within grid bounds
        self.ball_x = max(0, min(self.ball_x, self.grid_size - 1))
        self.ball_y = max(0, min(self.ball_y, self.grid_size - 1))
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True
            
        return self.get_state(), (reward_left, reward_right), self.done

    def render(self):
        """
        Visualize the grid with ball and paddles
        """
        grid = np.full((self.grid_size, self.grid_size), "0", dtype=str)
        grid[:, 0] = "|"
        grid[:, -1] = "|"
        grid[self.paddle_y_left, 0] = "L"
        grid[self.paddle_y_right, -1] = "R"
        grid[self.ball_y, self.ball_x] = "*"

        for row in grid:
            print(" ".join(row))
        print("\n")
    
    def get_score(self):
        """
        Gets the current scores

        :return (ing): current score of left agent and right agent
        """
        return (self.score_left, self.score_right)

def verify_unique_indices(env: PongEnv):
    unique_indices = set()
    duplicates = False
    grid_size = env.grid_size

    # Iterate over all possible values for ball position, velocity, and paddle position
    for ball_x in range(grid_size):
        for ball_y in range(grid_size):
            for ball_dx in [-1, 0, 1]:  # Horizontal velocity: left, neutral, right
                for ball_dy in [-1, 0, 1]:  # Vertical velocity: up, neutral, down
                    for paddle_y_left in range(grid_size):  # Left paddle position
                        for paddle_y_right in range(grid_size):  # Right paddle position
                            # Set the environment's state
                            env.ball_x = ball_x
                            env.ball_y = ball_y
                            env.paddle_y_left = paddle_y_left
                            env.paddle_y_right = paddle_y_right
                            env.ball_dx = ball_dx
                            env.ball_dy = ball_dy

                            # Get the index for the current state
                            state_index = env.get_state_index()

                            # Check if the index is already in the set
                            if state_index in unique_indices:
                                print(f"Duplicate index found: "
                                  f"Ball position ({ball_x}, {ball_y}), "
                                  f"Velocity ({ball_dx}, {ball_dy}), "
                                  f"Left paddle position {paddle_y_left}, "
                                  f"Right paddle position {paddle_y_right} -> State Index: {state_index}")
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

def check_max_state_index(env):
    """
    Checks if the maximum calculated state index exceeds the total number of states
    by using get_state_index() with maximum state values.
    Returns True if the maximum index is valid, False otherwise.
    """
    grid_size = env.grid_size
    
    # Save the current state of the environment
    current_ball_x = env.ball_x
    current_ball_y = env.ball_y
    current_paddle_y_left = env.paddle_y_left
    current_paddle_y_right = env.paddle_y_right
    current_ball_dx = env.ball_dx
    current_ball_dy = env.ball_dy

    # Set the state variables to their maximum values
    env.ball_x = grid_size - 1
    env.ball_y = grid_size - 1
    env.paddle_y_left = grid_size - 1
    env.paddle_y_right = grid_size - 1
    env.ball_dx = 1  # Maximum dx value
    env.ball_dy = 1  # Maximum dy value

    # Get the state index using the modified state
    max_index = env.get_state_index()

    # Restore the environment's state
    env.ball_x = current_ball_x
    env.ball_y = current_ball_y
    env.paddle_y_left = current_paddle_y_left
    env.paddle_y_right = current_paddle_y_right
    env.ball_dx = current_ball_dx
    env.ball_dy = current_ball_dy

    total_states = env.get_number_of_states()
    
    # Check if the calculated maximum index is within bounds
    if max_index >= total_states:
        print(f"Warning: The calculated maximum index ({max_index}) exceeds the total states ({total_states})!")
        return False
    else:
        print(f"Maximum index ({max_index}) is within the bounds of total states ({total_states}).")
        return True


if __name__ == '__main__':
    env = PongEnv(grid_size=10)
    state = env.reset()
    print("Initial state:", state)
    print("Total states: ", env.get_number_of_states())
    print("Total actions: ", env.get_number_of_actions())
    check_max_state_index(env)
    verify_unique_indices(env)

