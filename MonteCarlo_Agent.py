import numpy as np
from random import randint

class MonteCarlo:

	def __init__(self, environment, gamma=None, learning_rate=None, epsilon=0.1):
		self.epsilon = epsilon
		self.gamma = 1		# no discounting
		self.number_of_actions = environment.get_number_of_actions()
		self.number_of_states = environment.get_number_of_states()

		self.state = 0

		self.visits_table = np.zeros((self.number_of_states, self.number_of_actions))
		self.q_table = np.zeros((self.number_of_states, self.number_of_actions))

		self.trajectory = [[0, 0, 0]]	# will be reward, state, action

	def get_visited_states_num(self):

		state_vals = np.sum(self.visits_table, axis=1)
		return np.count_nonzero(state_vals, axis=0)

	def is_first_visit(self, state, action, index) -> bool:

		for i in range(len(self.trajectory)):
			if self.trajectory[i][1] == state and self.trajectory[i][2] == action:
				return index == i
		
		return False

	def update_q(self):

		print(self.trajectory)
		
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


	def epsilon_greedy(self, actions):
		greatest = np.argmax(actions)

		random_num = randint(1, 100)

		if self.epsilon * 100 <= random_num:
			return greatest
		else:
			return randint(1, len(actions)) - 1

	def select_action(self, state):
		# record current state in trajectory array
		self.trajectory[-1][1] = state

		# call epsilon greedy
		action = self.epsilon_greedy(self.q_table[state, ])

		# record action in trajectory array
		self.trajectory[-1][2] = (action)

		return action

	def update(self, state, reward):
		# record new state and reward in trajectory array
		self.trajectory.append([reward, state, -1])

	def clear_trajectory(self):
		self.trajectory = [[0, 0, 0]]