import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Tuple, Dict, Type, Union

def pretty_print_metrics(avg_rewards, avg_wins, percentage_visited_0_to_450, percentage_visited_450_to_900):
    print("\nExperiment Metrics:")
    print("====================")
    print(f"Average Rewards (last 30 episodes): {avg_rewards.mean():.2f}")
    print(f"Average Win Rate (last 30 episodes): {avg_wins * 100:.2f}%")
    print(f"Percentage of State-Action Pairs Visited (States 0 to 450): {percentage_visited_0_to_450:.2f}%")
    print(f"Percentage of State-Action Pairs Visited (States 450 to 900): {percentage_visited_450_to_900:.2f}%")
    print("====================\n")
    
def pretty_print_metrics_all_ep(avg_rewards, avg_wins, percentage_visited_0_to_450, percentage_visited_450_to_900):
    print("Experiment Metrics:")
    print("====================")
    print(f"Average Rewards (all episodes): {avg_rewards.mean():.2f}")
    print(f"Average Win Rate (all episodes): {avg_wins * 100:.2f}%")
    print(f"Percentage of State-Action Pairs Visited (States 0 to 450): {percentage_visited_0_to_450:.2f}%")
    print(f"Percentage of State-Action Pairs Visited (States 450 to 900): {percentage_visited_450_to_900:.2f}%")
    print("====================\n")

def moving_average(data, window_size=30):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_visit_percentage(agent_name, visit_count, save_path=None):
    visit_percentage = (visit_count / np.sum(visit_count)) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(visit_percentage, cmap='viridis', annot=False, fmt=".1f", xticklabels=[0, 1, 2])
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.title(f"{agent_name} State-Action Visit Percentage")
    if save_path:
        file_path = os.path.join(save_path, f"{agent_name.lower()}_visit_percentage.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def plot_winning_percentage(agent_labels: List[str], avg_wins: List[float], save_path=None):
    plt.figure(figsize=(10, 6))
    # Plotting each agent's winning percentage dynamically
    plt.bar(agent_labels, [win * 100 for win in avg_wins], color=sns.color_palette("Set2", len(agent_labels)))
    plt.ylabel("Mean Winning Percentage (%)")
    plt.title("Mean Winning Percentage Comparison")
    if save_path:
        file_path = os.path.join(save_path, "winning_percentage.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def plot_cumulative_return(avg_rewards: List[np.ndarray], agent_labels: List[str], save_path=None):
    plt.figure(figsize=(10, 6))
    for i, rewards in enumerate(avg_rewards):
        plt.plot(moving_average(rewards), label=agent_labels[i], linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Cumulative Return")
    plt.title("Mean Cumulative Return over Episodes")
    plt.legend()
    if save_path:
        file_path = os.path.join(save_path, "cumulative_return.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def plot_mean_visited_states_percentage(visit_counts: List[np.ndarray], agent_labels: List[str], save_path=None):
    plt.figure(figsize=(10, 6))
    visited_states = [np.sum(visit_count > 0) / visit_count.size * 100 for visit_count in visit_counts]
    plt.bar(agent_labels, visited_states, color=sns.color_palette("Set2", len(agent_labels)))
    plt.ylabel("Mean Percentage of Visited States (%)")
    plt.title("Mean Percentage of Visited States Comparison")
    if save_path:
        file_path = os.path.join(save_path, "mean_visited_states_percentages.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def plot_winning_percentage_over_episodes(win_statuses, agent_labels, save_path=None):
    """
    Plots the winning percentage over episodes for multiple agents based on win statuses.

    :param win_statuses: List of lists where each sublist contains win statuses (1 for win, 0 for loss) for each agent.
    :param agent_labels: List of agent labels corresponding to each list in win_statuses.
    :param save_path: Optional path to save the plot image.
    """
    plt.figure(figsize=(12, 6))

    # Plot each agent's winning percentage moving average
    for i, wins in enumerate(win_statuses):
        moving_avg = moving_average(wins, window_size=30)
        plt.plot(moving_avg, label=f"{agent_labels[i]} Winning Percentage (Moving Avg)", color=f"C{i}")

    plt.xlabel("Episodes")
    plt.ylabel("Winning Percentage")
    plt.title("Winning Percentage Over Episodes")
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid()

    if save_path:
        file_path = os.path.join(save_path, "winning_percentage_over_episodes.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()

def plot_mean_visited_states_per_action(visit_count, agent_name, save_path=None):
    visit_percentage = (visit_count / np.sum(visit_count)) * 100
    mean_visited_states = np.mean(visit_percentage, axis=0)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mean_visited_states)), mean_visited_states, color='teal', alpha=0.7)
    plt.xticks(ticks=range(len(mean_visited_states)), labels=[0, 1, 2])
    plt.xlabel("Actions")
    plt.ylabel("Mean Percentage of Visited States (%)")
    plt.title(f"{agent_name} - Mean Percentage of Visited States for Each Action")
    plt.grid()
    if save_path:
        file_path = os.path.join(save_path, f"{agent_name.lower()}_mean_visited_states.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def plot_state_action_distribution(visit_count, agent_name, save_path=None):
    plt.figure(figsize=(12, 6))
    visit_counts_per_action = np.sum(visit_count, axis=0)
    plt.bar(range(len(visit_counts_per_action)), visit_counts_per_action, tick_label=["Stay Still", "Move Up", "Move Down"], color='teal', alpha=0.7)
    plt.xlabel("Actions")
    plt.ylabel("Total Visit Count")
    plt.title(f"{agent_name} - Distribution of Visit Count per Action")
    plt.grid()
    if save_path:
        file_path = os.path.join(save_path, f"{agent_name.lower()}_state_action_distribution.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()

def plot_state_action_distribution_all(visit_counts, agent_names, save_path=None):
    num_agents = len(agent_names)
    actions = ["Stay Still", "Move Up", "Move Down"]
    num_actions = len(actions)

    # Calculate total visits for each action per agent
    total_visits = [np.sum(vc, axis=0) for vc in visit_counts]  # Sum over states
    x = np.arange(num_actions)  # x-axis positions for actions
    bar_width = 0.2  # Width of each bar

    plt.figure(figsize=(12, 6))
    for i, (agent_name, visits) in enumerate(zip(agent_names, total_visits)):
        plt.bar(
            x + i * bar_width, 
            visits, 
            width=bar_width, 
            label=agent_name, 
            alpha=0.7
        )

    plt.xlabel("Actions")
    plt.ylabel("Total Visit Count")
    plt.title("State-Action Visit Counts per Agent")
    plt.xticks(x + bar_width * (num_agents - 1) / 2, actions)  # Center action labels
    plt.legend(title="Agents")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save or display the plot
    if save_path:
        file_path = os.path.join(save_path, "all_state_action_visits.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def plot_state_visitation(all_V_t, agent_class_name, save_path=None):
    """
    Plots the state visitation percentage (V_t) over episodes for all agents combined.

    :param all_V_t: A list containing the V_t data for each agent, where each entry is a 1D numpy array of size (episodes,).
    :param agent_class_name: The name of the agent class (e.g., 'SARSA_0' or 'QLearningAgent') to label the plot.
    """
    # Combine V_t data from all agents
    combined_V_t = np.mean(all_V_t, axis=0)  # Average over all agents for each episode
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(combined_V_t)), combined_V_t, label=f'{agent_class_name} State Visitation', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Percentage of States Visited')
    plt.title(f'{agent_class_name} - State Visitation (V_t) Over Episodes (All Agents)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        file_path = os.path.join(save_path, f"{agent_class_name.lower()}_state_visualization.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()

def plot_agent_scores(agent_scores: np.ndarray, agent_name: str, save_path:str = None):
    """
    Plot the average scores for a given agent.

    :param agent_scores: np.ndarray - Array of scores for the agent across episodes.
    :param label: str - Label for the agent, used for plot title and legend.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(agent_scores, label=agent_name)
    plt.title(f"{agent_name} Scores over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    if save_path:
        file_path = os.path.join(save_path, f"{agent_name.lower()}_scores.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def plot_all_agents_scores(agent_scores: List[np.ndarray], agent_labels: List[str], save_path=None):
    plt.figure(figsize=(10, 6))
    for i, scores in enumerate(agent_scores):
        plt.plot(scores, label=agent_labels[i])
    plt.title("Agent Performance Comparison")
    plt.xlabel("Episodes")
    plt.ylabel("Average Score")
    plt.legend(loc='best')
    if save_path:
        file_path = os.path.join(save_path, "all_scores.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()
    
def smooth_data(data: np.ndarray, window_size: int = 10) -> np.ndarray:
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_all_agents_scores_smoothed(agent_scores: List[np.ndarray], agent_labels: List[str], save_path=None):
    plt.figure(figsize=(10, 6))
    for i, scores in enumerate(agent_scores):
        smoothed_scores = smooth_data(scores)
        plt.plot(smoothed_scores, label=agent_labels[i])
    plt.title("Agent Performance Comparison (Smoothed)")
    plt.xlabel("Episodes")
    plt.ylabel("Average Score")
    plt.legend(loc='best')
    if save_path:
        file_path = os.path.join(save_path, "all_scores_smoothed.png")
        plt.savefig(file_path)
    #plt.show()
    plt.close()