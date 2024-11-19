import numpy as np
from random import randint

class QLearning:
	
	def __init__(self, num_states, num_actions, gamma=0.9, learning_rate=0.1, epsilon=0.2):

		self.gamma = gamma
		self.learning_rate = learning_rate
		self.epsilon = epsilon

		self.number_of_states = num_states
		self.number_of_actions = num_actions

		self.q_table = np.zeros((self.number_of_states, self.number_of_actions))

		self.trajectory = [[0, 0, 0]]	# will be reward, state, action
		self.state_actn_pairs = {}
  
	def get_number_of_states(self):
		return self.number_of_states

	def get_visited_states_num(self) -> int:
		sum_visits = sum(list(self.state_actn_pairs.values()))
		return sum_visits

	def get_reached_state_action_pairs(self):
		table = np.zeros((self.number_of_states, self.number_of_actions))

		for entry in self.trajectory:
			table[entry[1]][entry[2]] = 1

		return table
	
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
		self.cur_state = state

		# call epsilon greedy
		action = self.epsilon_greedy(self.q_table[state, ])

		# record action in trajectory array
		self.trajectory[-1][2] = (action)
		if not((state,action) in self.state_actn_pairs):
					self.state_actn_pairs[(state,action)] = 1

		return action

	def update(self, new_state, reward):
		
		cur_state = self.trajectory[-1][1]
		prev_action = self.trajectory[-1][2]
		cur_q_value = self.q_table[cur_state][prev_action]
		max_q_possible = max(self.q_table[new_state])
		new_q = cur_q_value + self.learning_rate * (reward + self.gamma * max_q_possible - cur_q_value)
		self.q_table[cur_state][prev_action] = new_q

		self.trajectory.append([reward, new_state, -1])
	
	def clear_trajectory(self):
		self.trajectory = [[0, 0, 0]]
		self.state_actn_pairs1 = {}