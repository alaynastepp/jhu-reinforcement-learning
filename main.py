import numpy as np
import matplotlib.pyplot as plt
import argparse

#from basicPongEnv import PongEnv
#from exampleAgent import Agent
from alaynaEnv import PongEnv
from QLearningAgent import QLearningAgent
from testAgent import TestAgent
from pongVisualizer import PongVisualizer

AGENT_COUNT = 10
EPISODE_COUNT = 1000
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
        print(f"Episode: {episode + 1}, State: {new_state}, Reward: {reward}, Done: {game_end}")
        env.render()
        # Update agent's knowledge
        agent.update(state_index, action, reward, next_state_index)
        if visualizer:
            ball_x, ball_y, paddle_y, _, _ = env.get_state()
            visualizer.render((ball_x, ball_y), paddle_y)
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

def verify_get_state_index(env):
    unique_indices = set()
    duplicates = False
    grid_size = env.grid_size  # Assuming grid_size is the dimension of your environment

    # Iterate over all possible values for ball position, velocity, and paddle position
    for ball_x in range(grid_size):
        for ball_y in range(grid_size):
            for ball_dx in [-1, 0, 1]:   # Assuming velocities are only -1 or +1
                for ball_dy in [-1, 0, 1]:
                    for paddle_y in range(grid_size):

                        # Set the environment to this state
                        env.ball_x, env.ball_y = ball_x, ball_y
                        env.ball_dx, env.ball_dy = ball_dx, ball_dy
                        env.paddle_y = paddle_y

                        # Calculate the state index
                        state_index = env.get_state_index()

                        # Check for uniqueness of the state index
                        if state_index in unique_indices:
                            print(f"Duplicate index found for state: "
                                  f"Ball position ({ball_x}, {ball_y}), "
                                  f"Velocity ({ball_dx}, {ball_dy}), "
                                  f"Paddle position {paddle_y} -> State Index: {state_index}")
                            duplicates = True
                        else:
                            unique_indices.add(state_index)

    # Final summary
    if duplicates:
        print("There are duplicates in the state index calculations.")
    else:
        print("All state indices are unique. `get_state_index` logic appears correct.")
    print(f"Total unique states checked: {len(unique_indices)}")

if __name__ == '__main__':

	# Run SARSA agent
    #print("Training SARSA agent...")
    #avg_rewards = run_experiment(SARSA_0)

    # Run Q-Learning agent
    #print("Training Q-Learning agent...")
    #avg_rewards = run_trials(QLearningAgent)
    
    print("Training Test agent...")
    avg_rewards = run_trials(TestAgent)
    
    #verify_get_state_index(PongEnv())

	# TODO: output and save metrics
	
