import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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