class PongEnv:
    def __init__(self, grid_size=10, ball_dx=1, ball_dy=1, ball_x=None, ball_y=None):
        self.grid_size = grid_size
        self.initial_ball_dx = ball_dx
        self.initial_ball_dy = ball_dy
        self.initial_ball_x = ball_x if ball_x is not None else self.grid_size // 2 # if not specificed, ball starts in the center
        self.initial_ball_y = ball_y if ball_y is not None else self.grid_size // 2 # if not specificed, ball starts in the center

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
    
    def get_state_val(self):
        """
        Get the current state in form of an integer

        :return state (int): Current state encompassing information: ball position, paddle position, and velocity
        """

        # grid is width and height of 10, so values will be 0 through 9
        # each part of the state is single digits
        ball_dx_val = 0 if self.ball_dx > 0 else 1
        ball_dy_val = 0 if self.ball_dy > 0 else 1
        return self.ball_x + (self.ball_y * 10) + (self.paddle_y * 100) + (self.ball_dx * 1000) + (ball_dx_val * 10000) + (self.ball_dy * 100000) + (ball_dy_val * 1000000)

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
        # Ball position: 10 x 10 grid - 100 possible positions
        ball_positions = self.grid_size * self.grid_size  

        # Paddle position: 10 possible y-values
        paddle_positions = self.grid_size

        # Number of possible velocities for the ball (assuming -1, 0, 1 for dy, -1 and 1 for dx - ball can't go straight vertical so no 0)
        ball_velocity = 3 * 2 

        # Total number of states
        total_states = ball_positions * paddle_positions * ball_velocity
        
        #TODO - broke down the calculation so you can double check its the right total number of states
        #TODO - change to one-liner if correct: return (self.grid_size ** 2) * self.grid_size * 6  # ball positions * paddle positions * velocities
        
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
        raise NotImplementedError

    def render(self):
        """
        Visualize the grid with ball and paddle
        """
        raise NotImplementedError
    
    def get_terminal_states(self):
        """
        Gets all the possible terminal states that end the game

        :return (list): list of ints that represent terminal states
        """
        raise NotImplementedError

env = PongEnv(grid_size=10)
print("Total states: ", env.get_number_of_states())
print("Total actions: ", env.get_number_of_actions())