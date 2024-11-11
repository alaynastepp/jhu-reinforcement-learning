import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def pretty_print_metrics(avg_rewards, avg_wins, percentage_visited_0_to_450, percentage_visited_450_to_900):
    print("\nExperiment Metrics:")
    print("====================")
    print(f"Average Rewards (last 30 episodes): {avg_rewards.mean():.2f}")
    print(f"Average Win Rate (last 30 episodes): {avg_wins * 100:.2f}%")
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
    plt.show()

def plot_winning_percentage(agent1_label, avg_wins1, agent2_label, avg_wins2, agent3_label, avg_wins3, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.bar([agent1_label, agent2_label, agent3_label], [avg_wins1 * 100, avg_wins2 * 100, avg_wins3 * 100], color=['blue', 'orange', 'green'])
    plt.ylabel("Mean Winning Percentage (%)")
    plt.title("Mean Winning Percentage Comparison")
    if save_path:
        file_path = os.path.join(save_path, "winning_percentage.png")
        plt.savefig(file_path)
    plt.show()

def plot_cumulative_return(avg_rewards1, agent1_label, avg_rewards2, agent2_label, avg_rewards3, agent3_label, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(avg_rewards1), label=agent1_label, color='blue')
    plt.plot(moving_average(avg_rewards2), label=agent2_label, color='orange')
    plt.plot(moving_average(avg_rewards3), label=agent3_label, color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Mean Cumulative Return")
    plt.title("Mean Cumulative Return over Episodes")
    plt.legend()
    if save_path:
        file_path = os.path.join(save_path, "cumulative_return.png")
        plt.savefig(file_path)
    plt.show()

def plot_mean_visited_states_percentage(visit_count1, agent1_label, visit_count2, agent2_label, visit_count3, agent3_label, save_path=None):
    visited_states1 = np.sum(visit_count1 > 0) / visit_count1.size * 100
    visited_states2 = np.sum(visit_count2 > 0) / visit_count2.size * 100
    visited_states3 = np.sum(visit_count3 > 0) / visit_count3.size * 100
    plt.figure(figsize=(10, 6))
    plt.bar([agent1_label, agent2_label, agent3_label], [visited_states1, visited_states2, visited_states3], color=['blue', 'orange', 'green'])
    plt.ylabel("Mean Percentage of Visited States (%)")
    plt.title("Mean Percentage of Visited States Comparison")
    if save_path:
        file_path = os.path.join(save_path, "mean_visited_states_percentages.png")
        plt.savefig(file_path)
    plt.show()

def plot_winning_percentage_over_episodes(agent1_wins, agent1_label, agent2_wins, agent2_label, agent3_wins, agent3_label, save_path=None):
    plt.figure(figsize=(12, 6))
    agent1_moving_avg = moving_average(agent1_wins, window_size=30)
    agent2_moving_avg = moving_average(agent2_wins, window_size=30)
    agent3_moving_avg = moving_average(agent3_wins, window_size=30)
    plt.plot(agent1_moving_avg, label=f"{agent1_label} Winning Percentage (Moving Avg)", color='blue')
    plt.plot(agent2_moving_avg, label=f"{agent2_label} Winning Percentage (Moving Avg)", color='orange')
    plt.plot(agent3_moving_avg, label=f"{agent3_label} Winning Percentage (Moving Avg)", color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Winning Percentage")
    plt.title("Winning Percentage Over Episodes")
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid()
    if save_path:
        file_path = os.path.join(save_path, "winning_percentage_over_episodes.png")
        plt.savefig(file_path)
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()