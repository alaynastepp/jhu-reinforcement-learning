import numpy as np
import random

class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None, max_steps=1000, agent_side='right', random_start=True):
        self.grid_size = grid_size
        self.random_start = random_start
    
        if self.random_start:
            self.initial_ball_dx = random.choice([-1, 1])
            self.initial_ball_dy = random.choice([-1, 1])
            self.initial_ball_x = random.choice(range(1, self.grid_size-1))
            self.initial_ball_y = random.choice(range(1, self.grid_size-1))
        else:
            self.initial_ball_dx = ball_dx
            self.initial_ball_dy = ball_dy
            self.initial_ball_x = ball_x if ball_x is not None else self.grid_size // 2
            self.initial_ball_y = ball_y if ball_y is not None else self.grid_size // 2
        
        self.agent_side = agent_side  
        self.paddle_y = self.grid_size // 2
        
        self.score = 0
        self.done = False
        self.current_step = 0
        self.max_steps = max_steps

        # Automatically set the agent's position based on the side
        if self.agent_side == 'left':
            self.agent_position = 0  # Left side
            self.wall_position = self.grid_size - 1  # Right side
        else:
            self.agent_position = self.grid_size - 1  # Right side
            self.wall_position = 0  # Left side

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
        Convert the current state (ball position, paddle position, velocity, and paddle side) into a unique index.

        :return (int): The unique index representing the current state.
        """
        # Encode the ball's position
        ball_pos = self.ball_x * self.grid_size + self.ball_y

        # Encode the paddle's vertical position
        paddle_pos = self.paddle_y

        # Encode the ball's velocity
        ball_velocity = (self.ball_dx + 1) * 3 + (self.ball_dy + 1)

        # Encode the paddle side: 0 for left, 1 for right
        paddle_side = 1 if self.agent_side == "right" else 0

        # Combine all parts into a unique index
        return (
            ball_pos * (self.grid_size * 9 * 2) +  # Factor in the paddle side (2 states)
            paddle_pos * 9 * 2 +                  # Factor in the paddle side (2 states)
            ball_velocity * 2 +                   # Factor in the paddle side (2 states)
            paddle_side
        )
    
    def get_state(self):
        """
        Get the current state (ball position, paddle position, and velocity)
        
        :return state (list): Current state: ball position, paddle position, and velocity
        """
        return (self.ball_x, self.ball_y, self.paddle_y, self.ball_dx, self.ball_dy, self.agent_side)

    def get_number_of_states(self):
        """
        Number of possible states (based on ball position, paddle position, and ball velocity)
        
        :return (int): Total number of states
        """
        grid_size = self.grid_size
        num_ball_positions = grid_size * grid_size # ball_x and ball_y
        num_paddle_positions = grid_size  # paddle_y
        num_velocities = 3 * 3 # ball_dx and ball_dy (-1, 0, 1 for both)
        side_factor = 2  # Two sides: left or right
        
        total_states = num_ball_positions * num_paddle_positions * num_velocities * side_factor
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
        
        # ball bounces off the wall (left or right)
        if self.ball_x == 0 or self.ball_x == self.grid_size - 1:
            self.ball_dx *= -1 # reverse direction
        
        # ball bounces off the walls (top and bottom)
        if self.ball_y == 0 or self.ball_y == self.grid_size - 1:
            self.ball_dy *= -1 # reverse direction

        # ball reaches the right side 
        if self.ball_x == self.agent_position:
            if self.paddle_y == self.ball_y:
                self.score += 1
                reward = +5
                
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
        grid = np.full((self.grid_size, self.grid_size), "0", dtype=str)
        
        # Place the wall on the opposite side of the agent's side
        grid[:, self.wall_position] = "|"
        
        # Place the paddle on the agent's side
        grid[:, self.agent_position] = "x"
        
        grid[self.ball_y, self.ball_x] = "*"
        grid[self.paddle_y, self.agent_position] = "p"

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

        :return (int): current score
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
                        for agent_side in ['left', 'right']:
                            # Set the environment to this state
                            env.ball_x, env.ball_y = ball_x, ball_y
                            env.ball_dx, env.ball_dy = ball_dx, ball_dy
                            env.paddle_y = paddle_y
                            env.agent_side = agent_side

                            # Calculate the state index
                            state_index = env.get_state_index()

                            # Check for uniqueness of the state index
                            if state_index in unique_indices:
                                print(f"Duplicate index found: "
                                    f"Ball position ({ball_x}, {ball_y}), "
                                    f"Velocity ({ball_dx}, {ball_dy}), "
                                    f"Paddle position {paddle_y}, "
                                    f"Agent {agent_side} -> State Index: {state_index}")
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