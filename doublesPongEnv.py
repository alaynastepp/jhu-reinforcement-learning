import numpy as np

class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None, max_steps=1000):
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
    
    def get_number_of_states(self):
        """
        Number of possible states (based on ball position, paddle position, and ball velocity)
        
        :return (int): Total number of states
        """
        grid_size = self.grid_size
        num_ball_positions = grid_size * grid_size # ball_x and ball_y
        num_paddle_positions = grid_size  # paddle_y
        num_velocities = 3 * 3 # ball_dx and ball_dy (-1, 0, 1 for both)
        side_factor = 2  # Two sides: left and right
        
        total_states = num_ball_positions * num_paddle_positions * num_velocities * side_factor
        return total_states

    def get_state_index(self, agent_side: str):
        """
        Convert the current state (ball position, paddle position, and velocity) into a unique index.

        :return (int): The unique index representing the current state.
        """
        # Encode the ball's position
        ball_pos = self.ball_x * self.grid_size + self.ball_y

        # Encode the paddle's vertical position
        paddle_pos = self.paddle_y_right if agent_side == "right" else self.paddle_y_left

        # Encode the ball's velocity
        ball_velocity = (self.ball_dx + 1) * 3 + (self.ball_dy + 1)

        # Encode the paddle side: 0 for left, 1 for right
        paddle_side = 1 if agent_side == "right" else 0

        # Combine all parts into a unique index
        return (
            ball_pos * (self.grid_size * 9 * 2) +  # Factor in the paddle side (2 states)
            paddle_pos * 9 * 2 +                  # Factor in the paddle side (2 states)
            ball_velocity * 2 +                   # Factor in the paddle side (2 states)
            paddle_side
        )
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
                self.ball_dy = self.update_ball_angle(action_left, self.ball_dy)
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
                self.ball_dy = self.update_ball_angle(action_right, self.ball_dy)
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
    
    def update_ball_angle(self, action, ball_dy):
        """
        Update ball angle based on paddle movement.
        :param action (int): The current action.
        :param ball_dy (int): The current velocity of the ball in the y direction.
        : return (int): Updated velocity of the ball in the y direction.
        """
        if ball_dy < 0:  # Ball coming down
            if action == 1:  # Paddle moving up
                return 0
            elif action == 2:  # Paddle moving down
                return -1
            else:  # Paddle stationary
                return -1
        elif ball_dy > 0:  # Ball coming up
            if action == 1:  # Paddle moving up
                return 1
            elif action == 2:  # Paddle moving down
                return 0
            else:  # Paddle stationary
                return 1
        else:  # Ball coming straight
            if action == 1:  # Paddle moving up
                return -1
            elif action == 2:  # Paddle moving down
                return 1
            else:  # Paddle stationary
                return 0
    
    def render(self):
        """
        Visualize the grid with ball and paddles
        """
        grid = np.full((self.grid_size, self.grid_size), "0", dtype=str)
        grid[:, 0] = "|"
        grid[:, -1] = "|"
        grid[self.ball_y, self.ball_x] = "*"
        grid[self.paddle_y_left, 0] = "L"
        grid[self.paddle_y_right, -1] = "R"

        for row in grid:
            print(" ".join(row))
        print("\n")
    
    def get_score(self):
        """
        Gets the current scores

        :return (ing): current score of left agent and right agent
        """
        return (self.score_left, self.score_right)


def verify_unique_indices(env) -> bool:
    """
    Verify that all possible state indexes generated by get_state_index are unique.
    
    :param env: The environment object with state space parameters.
    :return: True if all state indexes are unique, False otherwise.
    """
    unique_indices = set()
    duplicates = False
    grid_size = env.grid_size
    sides = ["left", "right"]

    # Loop through all possible states
    for ball_x in range(grid_size):
        for ball_y in range(grid_size):
            for paddle_y in range(grid_size):
                for ball_dx in [-1, 0, 1]:
                    for ball_dy in [-1, 0, 1]:
                        for agent in sides:
                            # Set the environment's state variables
                            env.ball_x = ball_x
                            env.ball_y = ball_y
                            env.paddle_y_left = paddle_y if agent == "left" else env.paddle_y_left
                            env.paddle_y_right = paddle_y if agent == "right" else env.paddle_y_right
                            env.ball_dx = ball_dx
                            env.ball_dy = ball_dy

                            # Get the state index for the current agent
                            state_index = env.get_state_index(agent)

                            # Check for uniqueness of the state index
                            if state_index in unique_indices:
                                print(f"Duplicate index found: "
                                    f"Ball position ({ball_x}, {ball_y}), "
                                    f"Velocity ({ball_dx}, {ball_dy}), "
                                    f"Paddle position {paddle_y}, "
                                    f"Agent {agent} -> State Index: {state_index}")
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

