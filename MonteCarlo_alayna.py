import numpy as np

class MonteCarloAgent:
    """Monte Carlo agent for Pong"""

    def __init__(self, num_states: int, num_actions: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2) -> None:
        """
        Initializes the Monte Carlo Agent with parameters and the Q-table.

        :param num_states (int): Number of possible states in the environment.
        :param num_actions (int): Number of possible actions.
        :param alpha (float): Learning rate.
        :param gamma (float): Discount factor.
        :param epsilon (float): Exploration rate.
        """
        self.q_table = np.zeros((num_states, num_actions), dtype="float64")
        self.n = np.zeros((num_states, num_actions), dtype="float64")
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.number_of_states = num_states
        self.number_of_actions = num_actions
        self.visited_states = []
        self.episode = {} # will be state, action, reward
        self.state_actn_pairs = {}  # Track (state, action) visit counts within an episode
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0

    def get_number_of_states(self):
        return self.number_of_states

    def get_number_of_actions(self):
        return self.number_of_actions
    
    def get_visited_states_num(self) -> int:
        """
        Calculates the total number of visits to state-action pairs.

        :return (int): Sum of visits to each state-action pair.
        """
        sum_visits = sum(list(self.state_actn_pairs.values()))
        return sum_visits

    def e_greedy(self, actions):
        a_star_idx = np.argmax(actions)
        rng = np.random.default_rng()
        if self.epsilon <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def select_action(self, state):
        self.turn += 1
        self.state = state
        actions = self.q_table[state, ]
        action = self.e_greedy(actions)
        self.action = action
        return action

    def update_q(self):
        #get final state info
        f_stte = self.episode[self.turn-1][0]
        f_actn = self.episode[self.turn-1][1]
        f_rwd = self.episode[self.turn-1][2]
        G = f_rwd
        self.n[f_stte][f_actn] += 1
        n = self.n[f_stte][f_actn]
        self.q_table[f_stte][f_actn] += (1/n)*(G- self.q_table[f_stte][f_actn])
        
        #update visited states
        if (f_stte in self.visited_states) == False:
            self.visited_states.append(f_stte)
        
        #update G, backtracking from second to last state
        for i in range(self.turn-2,-1,-1):
            stte = self.episode[i][0]
            actn = self.episode[i][1]
            rwd = self.episode[i][2]
            G = self.gamma*G + rwd
            visits = self.state_actn_pairs[(stte, actn)]
            if visits > 1:
                self.state_actn_pairs[(stte, actn)] -= 1
                continue
            else:
                self.n[stte][actn] += 1
                n = self.n[stte][actn]
                self.q_table[stte][actn] += (1/n)*(G- self.q_table[stte][actn])
            
            #update visited states
            if (stte in self.visited_states) == False:
                self.visited_states.append(stte)
        
        # get # of visited states for current episode
        v_t = len(self.visited_states)
        return G, v_t
      
    def update(self, current_state, reward):
        self.episode[self.turn-1] = [current_state, self.action, reward]
        if (current_state, self.action) in self.state_actn_pairs:
            self.state_actn_pairs[(current_state, self.action)] = self.state_actn_pairs[(current_state, self.action)] +1
        else:
            self.state_actn_pairs[(current_state, self.action)] = 1
            
    def new_episode(self):
        self.episode = {}
        self.state_actn_pairs = {}
        self.state = 0
        self.next_state = 0
        self.reward = 0
        self.action = 0
        self.turn = 0
        
    def reset(self):
        self.visited_states = []
        self.new_episode()
        self.q_table = np.zeros((self.number_of_states, self.number_of_actions), dtype="float64")
        self.n = np.zeros((self.number_of_states, self.number_of_actions), dtype="float64")
        
    def clear_trajectory(self):
        """
        Resets the episode data after each episode ends.
        """
        self.episode.clear()
        self.state_actn_pairs.clear()
        self.turn = 0