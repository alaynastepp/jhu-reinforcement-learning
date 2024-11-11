import numpy as np


class SARSA_0:
    """RL SARSA-0 agent for Pong"""

    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        Initializes the SARSA agent with parameters and the Q-table.

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
        self.next_action = -1
        self.next_q = 0
        self.reward = 0
        self.action = 0
        self.state_actn_pairs = {}

    def get_number_of_states(self):
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

    def e_greedy(self, actions):
        """
        Selects an action based on the epsilon-greedy strategy, favoring the
        best-known action with a probability of (1 - epsilon) and random action
        selection otherwise.

        :param actions (list[float]): List of Q-values for the current state's actions.
        :return (int): Index of the selected action.
        """
        a = list(np.where(np.array(actions) == max(actions))[0])
        b = len(a)
        rng = np.random.default_rng()
        
        if self.epsilon <= rng.random():
            if b < 2:
                a_star_idx = np.argmax(actions)
                return a_star_idx
            else:
                idx = rng.integers(low=0, high=b)
                return a[idx]
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx
    
    def get_state_actn_visits(self):
        """
        Calculates the total number of visits to state-action pairs.

        :return (int): Sum of visits to each state-action pair.
        """
        sum_visits = sum(list(self.state_actn_pairs.values()))
        return sum_visits
        
    def select_action(self, state):
        """
        Selects an action based on epsilon-greedy policy.

        :param state index (int): Current state as an integer
        :return (int): Action to take (0: stay, 1: up, 2: down)
        """
        self.state = state
        if self.next_action < 0:
            actions = self.q_table[state, ]
            action = self.e_greedy(actions)
        else:
            action = self.next_action
        self.action = action
        if not((state,action) in self.state_actn_pairs):
            self.state_actn_pairs[(state,action)] = 1
        return action

    def set_future_action(self, new_state):
        actions = self.q_table[new_state, ]
        action = self.e_greedy(actions)
        self.next_q = self.q_table[new_state, action]
        self.next_action = action
        
        
    def update(self, new_state, reward):
        """
        Updates Q-table based on the agent's experience.

        :param new_state_index (int): New state index after taking action
        :param reward (float): Reward received
        """
        self.set_future_action(new_state)
        self.next_state = new_state
        q =  self.q_table[self.state, self.action]
        self.q_table[self.state,self.action]=q+self.alpha*(reward+self.gamma*self.next_q-q)
        
    def reset(self):
        """
        Resets the Q-table and other agent-specific parameters for a new episode.
        """
        self.q_table = np.zeros((self.number_of_states, self.number_of_actions), dtype="float64")
        self.state_actn_pairs = {}
        self.state = 0
        self.next_state = 0
        self.next_action = -1
        self.next_q = 0
        self.reward = 0
        self.action = 0
