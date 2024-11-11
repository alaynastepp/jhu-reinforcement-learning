import numpy as np

class QLearingAgent:
    """
    RL Q-Learning agent for Pong
    """

    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q = np.zeros((num_states, num_actions), dtype="float64")
        self.state_actn_pairs = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.number_of_states = num_states
        self.number_of_actions = num_actions

    def get_number_of_states(self):
        return self.number_of_states

    def get_number_of_actions(self):
        return self.number_of_actions

    def e_greedy(self, actions):
        a=list(np.where(np.array(actions)==max(actions))[0])
        b=len(a)
        if b < 2:
            a_star_idx = np.argmax(actions)
            return a_star_idx
        else:
            rng = np.random.default_rng()
            idx = rng.integers(low=0, high=b)
            return a[idx]
    
    def get_state_actn_visits(self):
        sum1 = sum(list(self.state_actn_pairs.values()))
        return sum1
        
    def select_action(self, state_index):
        """
        Selects an action based on epsilon-greedy policy.

        :param state index (int): Current state as an integer
        :return (int): Action to take (0: stay, 1: up, 2: down)
        """
        self.turn += 1
        self.state = state_index
        actions = self.q[state_index, ]
        action = self.e_greedy(actions)
        self.action = action
        if not((state_index,action) in self.state_actn_pairs):
            self.state_actn_pairs[(state_index,action)] = 1
            
        return action

    def update(self, new_state_index, reward):
        """
        Updates Q-table based on the agent's experience.

        :param new_state_index (int): New state index after taking action
        :param reward (float): Reward received
        """
        self.next_state = new_state_index
        q =  self.q[self.state, self.action]
        max_q = max(self.q[new_state_index, ])
        self.q[self.state,self.action]=q+self.alpha*(reward+self.gamma*max_q-q)
        f"Turn = {self.turn} \nQ = {self.q}"

    def reset(self):
        """
        Resets the Q-table and other agent-specific parameters for a new episode.
        """
        self.q = np.zeros((self.number_of_states, self.number_of_actions), dtype="float64")
        self.state_actn_pairs1 = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0