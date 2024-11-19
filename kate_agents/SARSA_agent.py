import numpy as np
from random import randint

class SARSA:

	def __init__(self, num_states, num_actions, gamma=0.9, learning_rate=0.1, epsilon=0.2):

		self.gamma = gamma
		self.learning_rate = learning_rate
		self.epsilon = epsilon

		self.number_of_states = num_states
		self.number_of_actions = num_actions

		self.q_table = np.zeros((self.number_of_states, self.number_of_actions))
		self.state_actn_pairs = {}
  
		self.trajectory = [[0, 0, 0]]	# will be reward, state, action
		self.cur_state = None
		self.prev_action = None
	
	def get_number_of_states(self):
		return self.number_of_states

	def get_visited_states_num(self):
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

		# if this is our first action
		if self.prev_action is None:
			self.prev_action = self.epsilon_greedy(self.q_table[state])

		# record action in trajectory array
		self.trajectory[-1][2] = (self.prev_action)
		if not((state,self.prev_action) in self.state_actn_pairs):
			self.state_actn_pairs[(state,self.prev_action)] = 1
		return self.prev_action

	def update(self, new_state, reward):
		# record new state and reward in trajectory array
		self.trajectory.append([reward, new_state, -1])

		next_action = self.epsilon_greedy(self.q_table[new_state, ])

		# get helpers and updated q value
		cur_q_value = self.q_table[self.cur_state][self.prev_action]
		next_action_q_value = self.q_table[new_state][next_action]
		new_q = cur_q_value + self.learning_rate * (reward + self.gamma * next_action_q_value - cur_q_value)
		self.q_table[self.cur_state][self.prev_action] = new_q

		# transition to new state and force us to take the action on the next iteration
		self.cur_state = new_state
		self.prev_action = next_action

		self.trajectory.append([reward, new_state, -1])

	def clear_trajectory(self):
		self.trajectory = [[0, 0, 0]]
		self.cur_state = None
		self.prev_action = None
		self.state_actn_pairs1 = {}