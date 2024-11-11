import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Dict, Type, Union
import argparse

from QLearning_alayna import QLearningAgent
from SARSA_alayna import SARSA_0
from perfectAgent import PerfectAgent
import metrics
from basicPongEnv import PongEnv
from pongVisualizer import PongVisualizer
from MonteCarlo_Agent import MonteCarlo
from SARSA_Agent import SARSA
from QLearning_Agent import QLearning

HERE = os.path.dirname(os.path.abspath(__file__))

AGENT_COUNT = 1
EPISODE_COUNT = 200
WINDOW_LENGTH = 30
EXP_STARTS = False
DEBUG = False

def log(val):
	if DEBUG:
		print(val)
  
METRICS_PATH = os.path.join(HERE, 'experiment1')

if METRICS_PATH and not os.path.exists(METRICS_PATH):
        os.makedirs(METRICS_PATH)
        
def generate_episode(episode: int, env: PongEnv, agent, visualizer=None) -> Tuple[List[float], np.ndarray, Tuple, bool]:
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
        log(f"Episode: {episode + 1}, State: {new_state}, Reward: {reward}, Done: {game_end}")
        if DEBUG:
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
        
    if agent is MonteCarlo:
        agent.update_q()
        agent.clear_trajectory()
  
    # return the result of the game
    return rewards, episode_visit_count, current_state, win, env.get_score()

def run_trials(agent_class: Type[Union[QLearningAgent, SARSA_0, PerfectAgent, MonteCarlo]], args) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_class (class): One of the following: SARSA_0, QLearningAgent, or MonteCarlo.
    :param alpha (float): Learning rate.
    :param gamma (float): Discount factor.
    :param epsilon (float): Exploration rate.
    :return Dict containing the following metrics:
            'avg_rewards': np.ndarray - Average rewards over all agents.
            'avg_wins': float - Overall win rate.
            'state_action_visit_count': np.ndarray - Visit counts for each state-action pair.
            'win_statuses': np.ndarray - Array of win status for each episode.
            'state_visit_percentages': List[float] - State visit percentages across episodes.
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
    all_wins = []
    total_wins = 0
    visit_count = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions())) #raw count of how many times a specific state-action pair has been visited across episodes
    all_V_t = [] #percentage of the total states that have been visited 
    
    reward_file_name = f"{agent_class.__name__}_episode_rewards.txt"
    with open(reward_file_name, "w") as reward_file:
        for i in range(AGENT_COUNT):
            if agent_class == PerfectAgent:
                agent = agent_class(environment) 
            else:
                agent = agent_class(environment.get_number_of_states(), environment.get_number_of_actions(), **params)
            # initialize arrays for keeping track of agent performance over time
            episode_rewards = []
            win_status = []
            V_t = np.zeros((EPISODE_COUNT,1))  # percent states visited per episode
            wins = 0
            for i in range(EPISODE_COUNT): 
                # play game
                rewards, episode_visit_count, final_state, win, score = generate_episode(i, environment, agent, visualizer=visualizer)
                episode_rewards.append(sum(rewards))
                win_status.append(1 if win else 0)
                wins += win
                if agent_class != PerfectAgent:
                    v_t = agent.get_visited_states_num()
                    V_t[i,0] = (v_t/agent.get_number_of_states())*100
                #agent.clear_trajectory() 
            
            all_rewards.append(episode_rewards)
            total_wins += wins
            all_wins.append(win_status)
            visit_count += episode_visit_count
            all_V_t.append(V_t.flatten())
            
            for reward in episode_rewards:
                reward_file.write(f"{reward}\n")
                
            if visualizer is not None:
                visualizer.close()
        
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_wins = total_wins / (AGENT_COUNT * EPISODE_COUNT)  # Calculate win rate
    
    # Calculate average rewards over the last 30 episodes
    avg_rewards_last_30 = np.mean([np.convolve(rewards, np.ones(30) / 30, mode='valid') for rewards in all_rewards], axis=0)

    # Calculate percentage of wins for the last 30 episodes
    recent_wins = np.mean([win_status[-30:] for win_status in all_wins], axis=0)
    avg_wins_last_30 = np.mean(recent_wins)  

    # Visit count for states 0 to 450 only
    visit_count_0_to_450 = visit_count[:450, :]
    percentage_visited_0_to_450 = np.sum(visit_count_0_to_450 > 0) / visit_count_0_to_450.size * 100  

    # Visit count for states 450 to 900
    visit_count_450_to_900 = visit_count[450:9000, :]
    percentage_visited_450_to_900 = np.sum(visit_count_450_to_900 > 0) / visit_count_450_to_900.size * 100 

    metrics.pretty_print_metrics(avg_rewards_last_30, avg_wins_last_30, percentage_visited_0_to_450, percentage_visited_450_to_900)
    
    #return avg_rewards, avg_wins, visit_count, np.mean(all_wins, axis=0), all_V_t
    return {
        'avg_rewards': avg_rewards,
        'avg_wins': avg_wins,
        'state_action_visit_count': visit_count,
        'win_statuses': np.mean(all_wins, axis=0),
        'state_visit_percentages': all_V_t
    }

def run_trials_with_hyperparams(agent_class: Type[Union[QLearningAgent, SARSA_0, PerfectAgent]], alpha_values: List[float], gamma_values: List[float], epsilon_values: List[float]) -> None:
    """
    Runs multiple trials with different hyperparameter values and identifies the best configuration.

    :param agent_class: The agent class to use for training.
    :param alpha_values: A list of possible alpha (learning rate) values.
    :param gamma_values: A list of possible gamma (discount factor) values.
    :param epsilon_values: A list of possible epsilon (exploration rate) values.
    """
    best_avg_reward = -np.inf
    best_params = None

    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                print(f"Training {agent_class.__name__} with alpha={alpha}, gamma={gamma}, epsilon={epsilon}...")
                
                metrics = run_trials(
                    agent_class, alpha=alpha, gamma=gamma, epsilon=epsilon
                )
                
                avg_reward = np.mean(metrics['avg_rewards'])

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_params = (alpha, gamma, epsilon)

    if best_params:
        print("\n" + "*" * 50)
        print(f"***** Best Parameters Found *****")
        print(f"* Alpha:    {best_params[0]:.4f}")
        print(f"* Gamma:    {best_params[1]:.4f}")
        print(f"* Epsilon:  {best_params[2]:.4f}")
        print(f"* Avg Reward: {best_avg_reward:.2f}")
        print("*" * 50 + "\n")
        
#TODO - remove! this just checks that all state index values are unique
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sarsa', action='store_true', help='if SARSA algorithm should be run')
    parser.add_argument('--monte', action='store_true', help='if Monte Carlo algorithm should be run')
    parser.add_argument('--qlearning', action='store_true', help='if Q-Learning algorithm should be run')
    parser.add_argument('--viz', action='store_true', help="if visualization is wanted")
    parser.add_argument('--gamma', help="the value to be used for gamma")
    parser.add_argument('--learningrate', help='the value to be used for learning rate')
    parser.add_argument('--epsilon', help='the value to be used for epsilon')
    
    args = parser.parse_args()
    
    #print("Running Perfect agent...")
    #perfect_metrics = run_trials(PerfectAgent, args=args)
    
    if args.monte:
        monte_rewards, monte_scores = run_trials(MonteCarlo, args=args)
  
    if args.sarsa:
        print("Training SARSA agent...")
        sarsa_metrics = run_trials(SARSA_0, args=args)
    
    if args.qlearning:
        print("Training Q-Learning agent...")
        qlearning_metrics = run_trials(QLearningAgent, args=args)

    # Plot cumulative returns
    metrics.plot_cumulative_return(
        avg_rewards1=perfect_metrics["avg_rewards"],
        agent1_label="Perfect",
        avg_rewards2=sarsa_metrics["avg_rewards"],
        agent2_label="SARSA",
        avg_rewards3=qlearning_metrics["avg_rewards"],
        agent3_label="Q-Learning",
        save_path=METRICS_PATH
    )
    
    # Plot state visitation
    metrics.plot_state_visitation(perfect_metrics["state_visit_percentages"], "Perfect", save_path=METRICS_PATH)
    metrics.plot_state_visitation(sarsa_metrics["state_visit_percentages"], "SARSA", save_path=METRICS_PATH)
    metrics.plot_state_visitation(qlearning_metrics["state_visit_percentages"], "Q-Learning", save_path=METRICS_PATH)

    # Plot visit percentage
    metrics.plot_visit_percentage(agent_name="Perfect", visit_count=perfect_metrics["state_action_visit_count"], save_path=METRICS_PATH)
    metrics.plot_visit_percentage(agent_name="SARSA", visit_count=sarsa_metrics["state_action_visit_count"], save_path=METRICS_PATH)
    metrics.plot_visit_percentage(agent_name="Q-Learning", visit_count=qlearning_metrics["state_action_visit_count"], save_path=METRICS_PATH)

    # Plot winning percentage
    metrics.plot_winning_percentage(
        agent1_label="Perfect",
        avg_wins1=perfect_metrics["avg_wins"],
        agent2_label="SARSA",
        avg_wins2=sarsa_metrics["avg_wins"],
        agent3_label="Q-Learning",
        avg_wins3=qlearning_metrics["avg_wins"],
        save_path=METRICS_PATH
    )

    # Plot winning percentage over episodes
    metrics.plot_winning_percentage_over_episodes(
        agent1_wins=perfect_metrics["win_statuses"],
        agent1_label="Perfect",
        agent2_wins=sarsa_metrics["win_statuses"],
        agent2_label="SARSA",
        agent3_wins=qlearning_metrics["win_statuses"],
        agent3_label="Q-Learning",
        save_path=METRICS_PATH
    )

    # Additional metrics plots
    metrics.plot_mean_visited_states_percentage(
        visit_count1=perfect_metrics["state_action_visit_count"],
        agent1_label="Perfect",
        visit_count2=sarsa_metrics["state_action_visit_count"],
        agent2_label="SARSA",
        visit_count3=qlearning_metrics["state_action_visit_count"],
        agent3_label="Q-Learning",
        save_path=METRICS_PATH
    )
    
    metrics.plot_mean_visited_states_per_action(visit_count=perfect_metrics["state_action_visit_count"], agent_name="Perfect", save_path=METRICS_PATH)
    metrics.plot_mean_visited_states_per_action(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA", save_path=METRICS_PATH)
    metrics.plot_mean_visited_states_per_action(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning", save_path=METRICS_PATH)

    metrics.plot_state_action_distribution(visit_count=perfect_metrics["state_action_visit_count"], agent_name="Perfect", save_path=METRICS_PATH)
    metrics.plot_state_action_distribution(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA", save_path=METRICS_PATH)
    metrics.plot_state_action_distribution(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning", save_path=METRICS_PATH)
    
    #verify_get_state_index(PongEnv())
    
    # Tune hyperparameters
    alpha_values = [0.01, 0.1, 0.5]  # Example learning rates
    gamma_values = [0.5, 0.9, 0.95]  # Example discount factors
    epsilon_values = [0.1, 0.2, 0.5]  # Example exploration rates

    # Run experiments for SARSA
    run_trials_with_hyperparams(SARSA_0, alpha_values, gamma_values, epsilon_values)

    # Run experiments for Q-Learning
    run_trials_with_hyperparams(QLearningAgent, alpha_values, gamma_values, epsilon_values)
