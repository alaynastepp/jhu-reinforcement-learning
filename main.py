import numpy as np
import matplotlib.pyplot as plt
import argparse

from basicPongEnv import PongEnv
from exampleAgent import Agent

AGENT_COUNT = 10
EPISODE_COUNT = 1000
WINDOW_LENGTH = 30
EXP_STARTS = False

def generate_episode(env, agent):
	current_state = env.reset()
	game_end = False
	rewards = []

	while not game_end:
		action = agent.select_action(current_state)
		new_state, reward, game_end = env.execute_action(action)
		agent.update(new_state, reward)
		current_state = new_state
		rewards.append(reward)

	# return the result of the game
	return rewards, current_state

def run_trials(agent_type):
	"""
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_type (str): One of the following: "SARSA", "QLearning"
	"""
	
	# TODO: establish metrics for each agent
	for i in range(AGENT_COUNT):
		environment = PongEnv()
		agent = Agent()

		# TODO: initialize arrays for keeping track of agent performance over time

		for i in range(EPISODE_COUNT):

			# play game
			rewards, final_state = generate_episode(environment, agent)
			
			# TODO: record metrics

			agent.clear_trajectory()

	# TODO: return arrays full of metrics averaged over all agents
	return None

if __name__ == '__main__':

	run_trials("")

	# TODO: output and save metrics
	
