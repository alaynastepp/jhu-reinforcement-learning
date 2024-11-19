import numpy as np

class QLearningAgent:
    """
    RL Q-Learning agent for Pong
    """

    def __init__(self, num_states: int, num_actions: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2) -> None:
        """
        Initializes the Q-Learning Agent with parameters and the Q-table.

        :param num_states (int): Number of possible states in the environment.
        :param num_actions (int): Number of possible actions.
        :param alpha (float): Learning rate.
        :param gamma (float): Discount factor.
        :param epsilon (float): Exploration rate.
        """
        self.q_table = np.zeros((num_states, num_actions), dtype="float64")
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.number_of_states = num_states
        self.number_of_actions = num_actions
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.state_actn_pairs = {}

    def get_number_of_states(self) -> int:
        """
        Returns the number of states in the environment.
        
        :return (int): Number of states.
        """
        return self.number_of_states

    def get_number_of_actions(self):
        """
        Returns the number of actions available to the agent.
        
        :return (int): Number of actions.
        """
        return self.number_of_actions
    
    def get_visited_states_num(self) -> int:
        """
        Calculates the total number of visits to state-action pairs.

        :return (int): Sum of visits to each state-action pair.
        """
        sum_visits = sum(list(self.state_actn_pairs.values()))
        return sum_visits

    def e_greedy(self, actions: list[float]) -> int:
        """
        Selects an action based on the epsilon-greedy strategy, favoring the
        best-known action with a probability of (1 - epsilon) and random action
        selection otherwise.

        :param actions (list[float]): List of Q-values for the current state's actions.
        :return (int): Index of the selected action.
        """
        a = list(np.where(np.array(actions) == max(actions))[0])
        b = len(a)
        if b < 2:
            a_star_idx = np.argmax(actions)
            return a_star_idx
        else:
            rng = np.random.default_rng()
            idx = rng.integers(low=0, high=b)
            return a[idx]
         
    def select_action(self, state_index: int) -> int:
        """
        Selects an action based on epsilon-greedy policy.

        :param state_index (int): Current state as an integer
        :return (int): Action to take (0: stay, 1: up, 2: down)
        """
        self.state = state_index
        actions = self.q_table[state_index, ]
        action = self.e_greedy(actions)
        self.action = action
        if not((state_index,action) in self.state_actn_pairs):
            self.state_actn_pairs[(state_index,action)] = 1
            
        return action

    def update(self, new_state_index: int, reward: float) -> None:
        """
        Updates Q-table based on the agent's experience.

        :param new_state_index (int): New state index after taking action
        :param reward (float): Reward received
        """
        self.next_state = new_state_index
        q = self.q_table[self.state, self.action]
        max_q = max(self.q_table[new_state_index, ])
        self.q_table[self.state,self.action]=q+self.alpha*(reward+self.gamma*max_q-q)

    def reset(self) -> None:
        """
        Resets the Q-table and other agent-specific parameters for a new episode.
        """
        self.q_table = np.zeros((self.number_of_states, self.number_of_actions), dtype="float64")
        self.state_actn_pairs = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        