import numpy as np
from random import randint

class SARSA:

    def __init__(self, num_states: int, num_actions: int, 
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        """
        Initializes the SARSA agent with parameters and the Q-table.

        :param num_states (int): Number of possible states in the environment.
        :param num_actions (int): Number of possible actions.
        :param alpha (float): Learning rate.
        :param gamma (float): Discount factor.
        :param epsilon (float): Exploration rate.
        """

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.number_of_states = num_states
        self.number_of_actions = num_actions

        self.q_table = np.zeros((self.number_of_states, self.number_of_actions))
        self.state_actn_pairs = {}
  
        self.trajectory = [[0, 0, 0]]	# will be reward, state, action
        self.state = 0
        self.prev_action = None
	
    def get_number_of_states(self) -> int:
        """
        Returns the number of states in the environment.

        :return (int): Number of states.
        """
        return self.number_of_states
	
    def get_number_of_actions(self) -> int:
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

    def get_reached_state_action_pairs(self) -> list[int]:
        """
        returns table representing the reached state action pairs in the current
		trajectory stored
		
		:return (numpy array[int]): matrix with 0 axis state and 1 axis action
        """
        table = np.zeros((self.number_of_states, self.number_of_actions))

        for entry in self.trajectory:
            table[entry[1]][entry[2]] = 1

        return table

    def epsilon_greedy(self, actions: list[float]) -> int:
        """
        Selects an action based on the epsilon-greedy strategy, favoring the
        best-known action with a probability of (1 - epsilon) and random action
        selection otherwise.

        :param actions (list[float]): List of Q-values for the current state's actions.
        :return (int): Index of the selected action.
        """
        greatest = np.argmax(actions)

        random_num = randint(1, 100)

        if self.epsilon * 100 <= random_num:
            return greatest
        else:
            return randint(1, len(actions)) - 1

    def select_action(self, state: int) -> int:	
        """
        Selects an action based on epsilon-greedy policy.

        :param state (int): Current state as an integer
        :return (int): Action to take (0: stay, 1: up, 2: down)
        """
        # record current state in trajectory array
        self.trajectory[-1][1] = state
        self.state = state

        # if this is our first action
        if self.prev_action is None:
            self.prev_action = self.epsilon_greedy(self.q_table[state])

        # record action in trajectory array
        self.trajectory[-1][2] = (self.prev_action)
        if not((state,self.prev_action) in self.state_actn_pairs):
            self.state_actn_pairs[(state,self.prev_action)] = 1
        return self.prev_action

    def update(self, new_state: int, reward: float):
        """
        Updates Q-table based on the agent's experience.

        :param new_state (int): New state index after taking action
        :param reward (float): Reward received
        """
        # record new state and reward in trajectory array
        self.trajectory.append([reward, new_state, -1])

        next_action = self.epsilon_greedy(self.q_table[new_state, ])

        # get helpers and updated q value
        cur_q_value = self.q_table[self.state][self.prev_action]
        next_action_q_value = self.q_table[new_state][next_action]
        new_q = cur_q_value + self.alpha * (reward + self.gamma * next_action_q_value - cur_q_value)
        self.q_table[self.state][self.prev_action] = new_q

        # transition to new state and force us to take the action on the next iteration
        self.state = new_state
        self.prev_action = next_action

        self.trajectory.append([reward, new_state, -1])

    def reset(self):
        """
        Resets the Q-table and other agent-specific parameters for a new episode.
        """
        self.trajectory = [[0, 0, 0]]
        self.state = None
        self.prev_action = None
        self.state_actn_pairs = {}