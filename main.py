import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from basicPongEnv import PongEnv
from pongVisualizer import PongVisualizer
from MonteCarlo_Agent import MonteCarlo
from SARSA_Agent import SARSA

AGENT_COUNT = 10
EPISODE_COUNT = 15
WINDOW_LENGTH = 30
EXP_STARTS = False

def generate_episode(episode, env, agent, visualizer=None):
	"""
	Play one episode in the environment using the agent and collect rewards.

	:param episode (int): Current episode number.
	:param env (PongEnv): The Pong environment.
	:param agent: The agent that interacts with the environment.
	:param visualizer: Optional visualizer to render each step.
	:return rewards (list): List of rewards collected in the episode.
	:return final_state (tuple): The final state after the episode ends.
	"""
	current_state = env.reset()
	# In generate_episode, after state reset:
	print(f"Initial ball position: {env.ball_x}, paddle position: {env.paddle_y}")
	time.sleep(1)
	game_end = False
	rewards = []
	while not game_end:
		# Agent selects an action based on the current state
		state_index = env.get_state_index()
		action = agent.select_action(state_index)
		# Environment executes the action and returns the new state, reward, and done flag
		new_state, reward, game_end = env.execute_action(action)
		next_state_index = env.get_state_index()
		# Current state of game
		print(f"Episode: {episode}, State: {new_state}, Reward: {reward}, Action Taken: {action}, Done: {game_end}")
		env.render()
		time.sleep(1)
		# Update agent's knowledge
		agent.update(next_state_index, reward)
		if visualizer:
			ball_x, ball_y, paddle_y, _, _ = env.get_state()
			visualizer.render((ball_x, ball_y), paddle_y)
			time.sleep(1)
		current_state = new_state
		rewards.append(reward)
	# return the result of the game
	return rewards, current_state

def run_trials(agent_class):
	"""
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_class (class): One of the following: "SARSA_0", "QLearningAgent"
	"""
	environment = PongEnv(grid_size=10)
	visualizer = PongVisualizer(grid_size=10, cell_size=60)
    
	# TODO: establish metrics for each agent
    
	all_rewards = []
	for i in range(AGENT_COUNT):
		agent = agent_class(environment) #(environment.get_number_of_states(), environment.get_number_of_actions())
		# TODO: initialize arrays for keeping track of agent performance over time
        
		episode_rewards = []
		for i in range(EPISODE_COUNT): 
			# play game
			rewards, final_state = generate_episode(i, environment, agent, visualizer)
			episode_rewards.append(sum(rewards))
			# TODO: record metrics
			#agent.clear_trajectory()
		print("EPISODE REWARDS ", episode_rewards)
		# TODO: return arrays full of metrics averaged over all agents
		all_rewards.append(episode_rewards)
		visualizer.close()
	print("ALL REWARDS ", all_rewards)
	return np.mean(all_rewards, axis=0)

if __name__ == '__main__':

	run_trials(SARSA)

	# TODO: output and save metrics
	
