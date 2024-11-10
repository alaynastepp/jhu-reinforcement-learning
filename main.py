import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from basicPongEnv import PongEnv
from pongVisualizer import PongVisualizer
from MonteCarlo_Agent import MonteCarlo
from SARSA_Agent import SARSA
from QLearning_Agent import QLearning

AGENT_COUNT = 10
EPISODE_COUNT = 2000
WINDOW_LENGTH = 30
EXP_STARTS = False
DEBUG = False

def log(val):
	if DEBUG:
		print(val)

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
	log(f"Initial ball position: {env.ball_x}, paddle position: {env.paddle_y}")
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
		log(f"Episode: {episode}, State: {new_state}, Reward: {reward}, Action Taken: {action}, Done: {game_end}")

		# visualize in command line
		if DEBUG:
			env.render()
		
		# Update agent's knowledge
		agent.update(next_state_index, reward)
		if visualizer:
			ball_x, ball_y, paddle_y, _, _ = env.get_state()
			visualizer.render((ball_x, ball_y), paddle_y)
		current_state = new_state
		rewards.append(reward)
	
	if agent is MonteCarlo:
		agent.update_q()
		agent.clear_trajectory()
	
	# return the result of the game
	return rewards, current_state, env.get_score()

def run_trials(agent_class, args):
	"""
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_class (class): One of the following: SARSA, QLearning, MonteCarlo
	"""

	environment = PongEnv(grid_size=10)
	if args.viz:
		visualizer = PongVisualizer(grid_size=10, cell_size=60)
	else:
		visualizer = None

	params = {"gamma": args.gamma, "learning_rate": args.learningrate, "epsilon": args.epsilon}
	params = {k:float(v) for k,v in params.items() if v is not None}

	print(f"Running trials for {agent_class} with non-default args {params}")
    
	all_rewards = []
	all_scores = []
	for i in range(AGENT_COUNT):
		agent = agent_class(environment, **params) #(environment.get_number_of_states(), environment.get_number_of_actions())
		# TODO: initialize arrays for keeping track of agent performance over time
        
		episode_rewards = []
		episode_scores = []
		for i in range(EPISODE_COUNT): 
			# play game
			rewards, final_state, score = generate_episode(i, environment, agent, visualizer)

			# record episode metrics
			episode_rewards.append(sum(rewards))
			episode_scores.append(score)

		log(f"EPISODE REWARDS {episode_rewards}")
		
		# record all episode metrics
		all_rewards.append(episode_rewards)
		all_scores.append(episode_scores)
		
		if visualizer is not None:
			visualizer.close()

	log(f"ALL REWARDS {all_rewards}")
	
	# return mean over all agents run
	return np.mean(all_rewards, axis=0), np.mean(all_scores, axis=0)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--sarsa', action='store_true', help='if SARSA algorithm should be run')
	parser.add_argument('--monte', action='store_true', help='if Monte Carlo algorithm should be run')
	parser.add_argument('--qlearning', action='store_true', help='if Q-Learning algorithm should be run')

	parser.add_argument('--viz', action='store_true', help="if visualization is wanted")

	parser.add_argument('--gamma', help="the value to be used for gamma")
	parser.add_argument('--learningrate', help='the value to be used for learning rate')
	parser.add_argument('--epsilon', help='the value to be used for epsilon')

	args = parser.parse_args()

	if args.sarsa:
		sarsa_rewards, sarsa_scores = run_trials(SARSA, args)
		plt.plot(range(EPISODE_COUNT), sarsa_scores, label='sarsa')
	if args.monte:
		monte_rewards, monte_scores = run_trials(MonteCarlo, args)
		plt.plot(range(EPISODE_COUNT), monte_scores, label='monte')
	if args.qlearning:
		q_rewards, q_scores = run_trials(QLearning, args)
		plt.plot(range(EPISODE_COUNT), q_scores, label='qlearning')

	# save plots to disk
	plt.title('Total Episode Score')
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.legend()
	plt.savefig("scores.png")
	plt.close()

	# TODO: output and save metrics
	
