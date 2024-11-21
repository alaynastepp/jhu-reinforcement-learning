import numpy as np
from random import randint

class MonteCarlo:

    def __init__(self, num_states: int, num_actions: int,
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        """
        Initializes the Monte Carlo Agent with parameters and the Q-table.

        :param num_states (int): Number of possible states in the environment.
        :param num_actions (int): Number of possible actions.
        :param alpha (float): Learning rate.
        :param gamma (float): Discount factor.
        :param epsilon (float): Exploration rate.
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha      # not used in implementation
        self.number_of_actions = num_actions
        self.number_of_states = num_states

        print("e ", epsilon)
        print('a ', alpha)
        print('g ', gamma)
        self.state = 0

        self.visits_table = np.zeros((self.number_of_states, self.number_of_actions))
        self.q_table = np.zeros((self.number_of_states, self.number_of_actions))

        self.trajectory = [[0, 0, 0]]	# will be reward, state, action
        self.turn = 0
	
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
        state_vals = np.sum(self.visits_table, axis=1)
        return np.count_nonzero(state_vals, axis=0)

    def is_first_visit(self, state: int, action: int, index: int) -> bool:
        """
        Returns boolean designating whether the state action pair existing at
        a certain part of the trajectory is the first visit

        :param state (int): Current state as an integer
        :param action (int): Index of an action
        :param index (int): index in the trajectory that is of interest
        :return (boolean): True if first visit, False otherwise
        """
        for i in range(len(self.trajectory)):
            if self.trajectory[i][1] == state and self.trajectory[i][2] == action:
                return index == i
		    
        return False

    def update_q(self):
        """
        Iterate through trajectory and update the q table values
        """

        # initialize variables
        G = 0

        # move through the trajectory list backwards
        for i in range(len(self.trajectory) - 2, -1, -1):

            if self.is_first_visit(self.trajectory[i][1], self.trajectory[i][2], i):
                # update visits table
                self.visits_table[self.trajectory[i][1]][self.trajectory[i][2]] += 1

                # update G
                G  = self.gamma * G + self.trajectory[i+1][0]

                # update q
                visits_num = self.visits_table[self.trajectory[i][1]][self.trajectory[i][2]]
                cur_q = self.q_table[self.trajectory[i][1]][self.trajectory[i][2]]
                new_q = cur_q + (1 / visits_num) * (G - cur_q)
                self.q_table[self.trajectory[i][1]][self.trajectory[i][2]] = new_q


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

        :param state_index (int): Current state as an integer
        :return (int): Action to take (0: stay, 1: up, 2: down)
        """
        self.turn +=1

        # record current state in trajectory array
        self.trajectory[-1][1] = state

        # call epsilon greedy
        action = self.epsilon_greedy(self.q_table[state, ])

        # record action in trajectory array
        self.trajectory[-1][2] = (action)

        return action

    def update(self, state: int, reward: float):
        """
        Updates trajectory based on the agent's experience.

        :param new_state (int): New state index after taking action
        :param reward (float): Reward received
        """
        # record new state and reward in trajectory array
        self.trajectory.append([reward, state, -1])

    def reset(self):
        """
        Resets the Q-table and other agent-specific parameters for a new episode.
        """
        self.trajectory = [[0, 0, 0]]
        self.visits_table = np.zeros((self.number_of_states, self.number_of_actions))
        self.state = 0
        self.turn = 0
  
    def clear_trajectory(self):
        """
        Resets the episode data after each episode ends.
        """
        self.trajectory = [[0, 0, 0]]