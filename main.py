import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

#from basicPongEnv import PongEnv
#from exampleAgent import Agent
from alaynaEnv import PongEnv
from QLearning_agent import QLearingAgent
from testAgent import TestAgent
from pongVisualizer import PongVisualizer

AGENT_COUNT = 10
EPISODE_COUNT = 100
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
    win = False
    episode_visit_count = np.zeros((env.get_number_of_states(), env.get_number_of_actions()))  
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
        rewards.append(reward)
        episode_visit_count[state_index, action] += 1
        if game_end and reward > 0:
            win = True
        # Update agent's knowledge
        agent.update(next_state_index, reward)
        if visualizer:
            ball_x, ball_y, paddle_y, _, _ = env.get_state()
            visualizer.render((ball_x, ball_y), paddle_y)
        current_state = new_state
    # return the result of the game
    return rewards, episode_visit_count, current_state, win

def run_trials(agent_class):
    """
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_class (class): One of the following: "SARSA_0", "QLearningAgent"
	"""
    environment = PongEnv(grid_size=10)
    visualizer = PongVisualizer(grid_size=10, cell_size=60)
    # TODO: establish metrics for each agent
    all_rewards = []
    all_wins = []
    total_wins = 0
    visit_count = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions()))
    for i in range(AGENT_COUNT):
        if agent_class == QLearingAgent:
            agent = agent_class(environment.get_number_of_states(), environment.get_number_of_actions())
        elif agent_class == TestAgent:
            agent = agent_class(environment) 
        # TODO: initialize arrays for keeping track of agent performance over time
        episode_rewards = []
        win_status = []
        wins = 0
        for i in range(EPISODE_COUNT): 
            # play game
            rewards, episode_visit_count, final_state, win = generate_episode(i, environment, agent) #, visualizer=visualizer
            episode_rewards.append(sum(rewards))
            win_status.append(1 if win else 0)
            wins += win
            # TODO: record metrics
            #agent.clear_trajectory()
        print("EPISODE REWARDS ", episode_rewards)
        # TODO: return arrays full of metrics averaged over all agents
        all_rewards.append(episode_rewards)
        total_wins += wins
        all_wins.append(win_status)
        visit_count += episode_visit_count
        visualizer.close()
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_wins = total_wins / (AGENT_COUNT * EPISODE_COUNT)  # Calculate win rate
    
    # Calculate average rewards over the last 30 episodes
    avg_rewards_last_30 = np.mean([np.convolve(rewards, np.ones(30) / 30, mode='valid') for rewards in all_rewards], axis=0)

    # Calculate percentage of wins for the last 30 episodes
    recent_wins = np.mean([win_status[-30:] for win_status in all_wins], axis=0)
    avg_wins_last_30 = np.mean(recent_wins)  

    # Visit count for states 0 to 450 only
    visit_count_0_to_63 = visit_count[:450, :]
    percentage_visited_0_to_63 = np.sum(visit_count_0_to_63 > 0) / visit_count_0_to_63.size * 100  

    # Visit count for states 450 to 900
    visit_count_64_to_127 = visit_count[450:9000, :]
    percentage_visited_64_to_127 = np.sum(visit_count_64_to_127 > 0) / visit_count_64_to_127.size * 100 

    pretty_print_metrics(avg_rewards_last_30, avg_wins_last_30, percentage_visited_0_to_63, percentage_visited_64_to_127)

    return avg_rewards, avg_wins, visit_count, np.mean(all_wins, axis=0)

def pretty_print_metrics(avg_rewards, avg_wins, percentage_visited_0_to_63, percentage_visited_64_to_127):
    print("\nExperiment Metrics:")
    print("====================")
    print(f"Average Rewards (last 30 episodes): {avg_rewards.mean():.2f}")
    print(f"Average Win Rate (last 30 episodes): {avg_wins * 100:.2f}%")
    print(f"Percentage of State-Action Pairs Visited (States 0 to 450): {percentage_visited_0_to_63:.2f}%")
    print(f"Percentage of State-Action Pairs Visited (States 450 to 900): {percentage_visited_64_to_127:.2f}%")
    print("====================\n")

def moving_average(data, window_size=30):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
 
def plot_visit_percentage(agent_name, visit_count):
    visit_percentage = (visit_count / np.sum(visit_count)) * 100  # Calculate visit percentage
    plt.figure(figsize=(10, 8))
    sns.heatmap(visit_percentage, cmap='viridis', annot=False, fmt=".1f", xticklabels=[0, 1, 2])
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.title(f"{agent_name} State-Action Visit Percentage")
    plt.show()
    
    
def plot_winning_percentage(avg_wins_sarsa, avg_wins_qlearning):
    plt.figure(figsize=(8, 6))
    plt.bar(["Perfect Agent", "Q-Learning"], [avg_wins_sarsa * 100, avg_wins_qlearning * 100], color=['blue', 'orange'])
    plt.ylabel("Mean Winning Percentage (%)")
    plt.title("Mean Winning Percentage Comparison")
    plt.show()

def plot_cumulative_return(avg_rewards_sarsa, avg_rewards_qlearning):
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(avg_rewards_sarsa), label="Perfect Agent")
    plt.plot(moving_average(avg_rewards_qlearning), label="Q-Learning (Off-policy)")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Cumulative Return")
    plt.title("Mean Cumulative Return over Episodes")
    plt.legend()
    plt.show()

def plot_mean_visited_states_old(visit_count_sarsa, visit_count_qlearning):
    visited_states_sarsa = np.sum(visit_count_sarsa > 0) / visit_count_sarsa.size * 100
    visited_states_qlearning = np.sum(visit_count_qlearning > 0) / visit_count_qlearning.size * 100
    plt.figure(figsize=(8, 6))
    plt.bar(["Perfect Agent", "Q-Learning"], [visited_states_sarsa, visited_states_qlearning], color=['green', 'purple'])
    plt.ylabel("Mean Percentage of Visited States (%)")
    plt.title("Mean Percentage of Visited States Comparison")
    plt.show()
    
def plot_winning_percentage_over_episodes(sarsa_wins, qlearning_wins):
    plt.figure(figsize=(12, 6))
    
    # Calculate the moving average
    sarsa_moving_avg = moving_average(sarsa_wins, window_size=30)
    qlearning_moving_avg = moving_average(qlearning_wins, window_size=30)

    # Plot the winning percentages
    plt.plot(sarsa_moving_avg, label="Perfect Agent Winning Percentage (Moving Avg)", color='blue')
    plt.plot(qlearning_moving_avg, label="Q-Learning Winning Percentage (Moving Avg)", color='orange')
    
    plt.xlabel("Episodes")
    plt.ylabel("Winning Percentage")
    plt.title("Winning Percentage Over Episodes")
    plt.legend()
    plt.ylim(0, 1)  # Limit y-axis to show percentages (0% to 100%)
    plt.grid()
    plt.show()

def plot_mean_visited_states(visit_count, agent_name):
    visit_percentage = (visit_count / np.sum(visit_count)) * 100  # Calculate visit percentage
    mean_visited_states = np.mean(visit_percentage, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mean_visited_states)), mean_visited_states, color='teal', alpha=0.7)
    plt.xticks(ticks=range(len(mean_visited_states)), labels=[0, 1, 2])
    plt.xlabel("Actions")
    plt.ylabel("Mean Percentage of Visited States (%)")
    plt.title(f"{agent_name} - Mean Percentage of Visited States for Each Action")
    plt.grid()
    plt.show()

def plot_state_action_distribution(visit_count, agent_name):
    plt.figure(figsize=(12, 6))
    visit_counts_per_action = np.sum(visit_count, axis=0)
    plt.bar(range(len(visit_counts_per_action)), visit_counts_per_action, tick_label=["Stay Still", "Move Up", "Move Down"], color='teal', alpha=0.7)
    plt.xlabel("Actions")
    plt.ylabel("Total Visit Count")
    plt.title(f"{agent_name} - Distribution of Visit Count per Action")
    plt.grid()
    plt.show()
    
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
    
    print("Training Test agent...")
    test_rewards, test_wins, test_visit_count, test_win_status = run_trials(TestAgent)

    # Run Q-Learning agent
    print("Training Q-Learning agent...")
    qlearning_rewards, qlearning_wins, qlearning_visit_count, qlearning_win_status = run_trials(QLearingAgent)
    
    plot_cumulative_return(test_rewards, qlearning_rewards)
    plot_visit_percentage("test", test_visit_count)
    plot_visit_percentage("Q-Learning", qlearning_visit_count)
    plot_winning_percentage(test_wins, qlearning_wins)
    plot_winning_percentage_over_episodes(test_win_status, qlearning_win_status)
    plot_mean_visited_states_old(test_visit_count, qlearning_visit_count)
    #plot_mean_visited_states(test_visit_count, "test")
    #plot_mean_visited_states(qlearning_visit_count, "Q-Learning")
    plot_state_action_distribution(test_visit_count, "test")
    plot_state_action_distribution(qlearning_visit_count, "Q-Learning")
    
    #verify_get_state_index(PongEnv())

	# TODO: output and save metrics
	
