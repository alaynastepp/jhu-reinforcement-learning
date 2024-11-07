
class Agent:

	def __init__(self, num_actions, num_states, gamma=0.9, learning_rate=0.1, epsilon=0.2):
		raise NotImplementedError

	def select_action(self, state):
		"""
		Runs its internal algorithm to choose the action

		:param state (int): the integer representation of the current state
		:return action (int): the action the Agent wants to take
		"""
		raise NotImplementedError

	def update(self, state, action, reward, next_state_index):
		"""
		Runs its internal algorithm to update the q table or do whatever is necessary

		:param state (int): the new state since the last action was taken
		:param reward (int): reward given for the last action taken
		"""
		raise NotImplementedError