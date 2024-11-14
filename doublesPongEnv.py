import numpy as np

class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None, max_steps=10000):
        self.grid_size = grid_size
        self.initial_ball_dx = ball_dx
        self.initial_ball_dy = ball_dy
        self.initial_ball_x = ball_x if ball_x is not None else self.grid_size // 2
        self.initial_ball_y = ball_y if ball_y is not None else self.grid_size // 2
        self.current_step = 0
        self.max_steps = max_steps
        self.reset()

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
        # max ball position is 9,9 
        max_ball_pos = (self.grid_size - 1) * 10 + (self.grid_size - 1)

        # max paddle position is 9
        max_paddle = self.grid_size - 1

        # with explanation above in get_state_index
        # ball_dx = 1, ball_dy = 1
        max_ball_velocity = (1 + 1) * 3 + (1 + 1)
        
        return max_ball_pos * 90 + max_paddle * 9 + max_ball_velocity

    def get_state_index(self, agent="left"):
        """
        Convert the current state (ball position, paddle positions, and velocity) into a unique index,
        based on the specified agent.

        :param agent: The agent to compute the state index for ("left" or "right").
        :return (int): The unique index representing the current state for the specified agent.
        """
        # Encode the ball's position on a 2D grid (like coordinates)
        ball_pos = self.ball_x * self.grid_size + self.ball_y
        
        # Encode the vertical positions of both paddles (left and right)
        paddle_left_pos = self.paddle_y_left
        paddle_right_pos = self.paddle_y_right
        
        # Encode the ball’s velocity (dx and dy) into a unique index
        ball_velocity = (self.ball_dx + 1) * 3 + (self.ball_dy + 1)  # Encode dx and dy values

        # Depending on the agent specified, select the correct paddle position
        if agent == "left":
            paddle_pos = paddle_left_pos
        elif agent == "right":
            paddle_pos = paddle_right_pos
        else:
            raise ValueError("Invalid agent. Use 'left' or 'right'.")

        # Combine them all
        # Using different factors for ball_pos, paddle_pos, and ball_velocity
        # ensures that each combination has a unique state index.
        return (ball_pos * 90) + (paddle_pos * 9) + ball_velocity
    
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

if __name__ == '__main__':
    env = PongEnv(grid_size=10)
    state = env.reset()
    print("Initial state:", state)
    print("Total actions:", env.get_number_of_actions())

    for _ in range(5):
        # Sample actions (0 = stay, 1 = up, 2 = down)
        action_left = np.random.choice([0, 1, 2])
        action_right = np.random.choice([0, 1, 2])
        state, reward, done = env.execute_action(action_left, action_right)
        env.render()
        print("State:", state)
        print("Reward:", reward)
        print("Done:", done)
        if done:
            break
