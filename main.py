import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from typing import List, Tuple, Dict, Type, Union
import argparse
import pickle

from alayna_files.QLearning_agent import QLearningAgent
from alayna_files.SARSA_agent import SARSA_0
from alayna_files.perfect_agent import PerfectAgent
from alayna_files.MonteCarlo_agent import MonteCarloAgent
import metrics
from pongEnv import PongEnv
from pongVisualizer import PongVisualizer
from kate_files.MonteCarlo_agent import MonteCarlo
from kate_files.SARSA_agent import SARSA
from kate_files.QLearning_agent import QLearning

HERE = os.path.dirname(os.path.abspath(__file__))

AGENT_COUNT = 10
EPISODE_COUNT = 1000
WINDOW_LENGTH = 30
EXP_STARTS = False
DEBUG = False
METRICS_PATH = os.path.join(HERE, 'experiment1')
TRAINED_AGENTS_PATH = os.path.join(HERE, 'trained_agents')

def log(val):
	if DEBUG:
		print(val)

if METRICS_PATH:
    if not os.path.exists(METRICS_PATH):
        os.makedirs(METRICS_PATH)
    else:
        shutil.rmtree(METRICS_PATH)
        os.makedirs(METRICS_PATH)
        
if TRAINED_AGENTS_PATH and not os.path.exists(TRAINED_AGENTS_PATH):
        os.makedirs(TRAINED_AGENTS_PATH)
        
def generate_episode(episode: int, env: PongEnv, agent: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], visualizer=None) -> Tuple[List[float], np.ndarray, Tuple, bool]:
    """
    Play one episode in the environment using the agent and collect rewards.

    :param episode (int): Current episode number.
    :param env (PongEnv): The Pong environment.
    :param agent: The agent that interacts with the environment.
    :param visualizer: Optional visualizer to render each step.
    :return rewards (List[float]): A list of rewards collected during the episode.
    :return episode_visit_count (np.ndarray): A 2D array tracking state-action visit counts for the episode.
    :return current_state (Tuple): The final state of the environment after the episode ends.
    :return win (bool): True if the agent wins the episode, False otherwise.
    :return env_score (int): The score achieved in the environment after the episode ends.
    """
    current_state = env.reset()
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
        
        if game_end and (sum(rewards) > 0):
            win = True

        # Update agent's knowledge
        agent.update(next_state_index, reward)
        if visualizer:
            ball_x, ball_y, paddle_y, _, _, agent_side = env.get_state()
            visualizer.render_static((ball_x, ball_y), paddle_y, agent_side)
        current_state = new_state
        
    if type(agent) is MonteCarlo or type(agent) is MonteCarloAgent:
        agent.update_q()
        agent.clear_trajectory()
  
    # return the result of the game
    return rewards, episode_visit_count, current_state, win, env.get_score()

def run_trials(agent_class: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], args : argparse.Namespace) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_class (class): One of the following: SARSA_0, QLearningAgent, or MonteCarlo.
    :param args (argparse.Namespace): Parsed arguments from argparse containing parameters such as alpha, gamma, and epsilon.
    :return Dict containing the following metrics:
            'avg_rewards': np.ndarray - Average rewards over all agents.
            'avg_wins': float - Overall win rate.
            'avg_scores': np.ndarray - Average scores over all agents.
            'state_action_visit_count': np.ndarray - Visit counts for each state-action pair.
            'win_statuses': np.ndarray - Array of win status for each episode.
            'state_visit_percentages': List[float] - State visit percentages across episodes.
	"""
    if args.left:
        agent_side="left"
        environment = PongEnv(grid_size=10, agent_side="left")
    elif args.right:
        agent_side="right"
        environment = PongEnv(grid_size=10, agent_side="right")
    else:
        environment = PongEnv(grid_size=10)
        agent_side=environment.agent_side
        
    if args.viz:
        visualizer = PongVisualizer(grid_size=10, cell_size=60)
    else:
        visualizer = None
        
    params = {"gamma": args.gamma, "learning_rate": args.learningrate, "epsilon": args.epsilon}
    params = {k:float(v) for k,v in params.items() if v is not None}
    print(f"Running trials for {agent_class} with non-default args {params}")

    all_rewards = []
    all_scores = []
    all_wins = []
    total_wins = 0
    visit_count = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions())) #raw count of how many times a specific state-action pair has been visited across episodes
    all_V_t = [] #percentage of the total states that have been visited 
    
    for a in range(AGENT_COUNT):
        if agent_class == PerfectAgent:
            agent = agent_class(environment) 
        else:
            agent = agent_class(environment.get_number_of_states(), environment.get_number_of_actions(), **params)
        # initialize arrays for keeping track of agent performance over time
        episode_rewards = []
        episode_scores = []
        win_status = []
        V_t = np.zeros((EPISODE_COUNT,1))  # percent states visited per episode
        wins = 0
        for i in range(EPISODE_COUNT): 
            # play game
            rewards, episode_visit_count, final_state, win, score = generate_episode(i, environment, agent, visualizer=visualizer)
            episode_rewards.append(sum(rewards))
            episode_scores.append(score)
            win_status.append(1 if win else 0)
            wins += win
            visit_count += episode_visit_count
            if agent_class != PerfectAgent:
                v_t = agent.get_visited_states_num()
                V_t[i,0] = (v_t/agent.get_number_of_states())*100
        
        all_rewards.append(episode_rewards)
        all_scores.append(episode_scores)
        total_wins += wins
        all_wins.append(win_status)
        all_V_t.append(V_t.flatten())
            
        if visualizer is not None:
            visualizer.close()
        
        # save off trained agent
        if args.save:
            save_agent(agent, os.path.join(TRAINED_AGENTS_PATH, f'{agent_side}_trained_{str(agent_class.__name__)}_{a}.pkl'))
        
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
        'avg_rewards': np.mean(all_rewards, axis=0),
        'avg_wins': total_wins / (AGENT_COUNT * EPISODE_COUNT),  # Calculate win rate,
        'avg_scores': np.mean(all_scores, axis=0),
        'state_action_visit_count': visit_count,
        'win_statuses': np.mean(all_wins, axis=0),
        'state_visit_percentages': all_V_t
    }

def run_trials_with_hyperparams(agent_class: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], alpha_values: List[float], gamma_values: List[float], epsilon_values: List[float], args) -> None:
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
                    agent_class, alpha=alpha, gamma=gamma, epsilon=epsilon, args=args
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

def save_agent(agent, path):
    """
    Save the Q-table of an agent to a file using pickle.

    :param agent: The agent whose Q-table is to be saved.
    :param path (str): The file path where the Q-table will be saved.
    """
    with open(path, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print(f"Agent saved to {path}")

def load_agent(agent_class, filename, *args, **kwargs):
    """
    Load an agent's Q-table from a file and initialize the agent.

    :param agent_class: The class of the agent to be initialized (e.g., QLearningAgent).
    :param filename (str): The file path from which the Q-table will be loaded.
    :param *args: Additional positional arguments for initializing the agent.
    :param **kwargs: Additional keyword arguments for initializing the agent.
    :return: An instance of the agent with the loaded Q-table.
    """
    # Initialize a new agent
    agent = agent_class(*args, **kwargs)
    # Load the saved Q-table
    with open(filename, 'rb') as f:
        agent.q_table = pickle.load(f)
    print(f"Agent loaded from {filename}")
    return agent


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sarsa', action='store_true', help='if SARSA algorithm should be run')
    parser.add_argument('--monte', action='store_true', help='if Monte Carlo algorithm should be run')
    parser.add_argument('--qlearning', action='store_true', help='if Q-Learning algorithm should be run')
    parser.add_argument('--sarsa_kate', action='store_true', help='if SARSA algorithm should be run')
    parser.add_argument('--monte_kate', action='store_true', help='if Monte Carlo algorithm should be run')
    parser.add_argument('--qlearning_kate', action='store_true', help='if Q-Learning algorithm should be run')
    parser.add_argument('--viz', action='store_true', help="if visualization is wanted")
    parser.add_argument('--plot', action='store_true', help="if plotting is wanted")
    parser.add_argument('--gamma', help="the value to be used for gamma")
    parser.add_argument('--learningrate', help='the value to be used for learning rate')
    parser.add_argument('--epsilon', help='the value to be used for epsilon')
    parser.add_argument('--left', action='store_true', help='if the agent is on the left side')
    parser.add_argument('--right', action='store_true', help='if the agent is on the right side')
    parser.add_argument('--save', action='store_true', help='if the agent should get saved off as a trained agent')
    
    args = parser.parse_args()

    #print("Running Perfect agent...")
    #perfect_metrics = run_trials(PerfectAgent, args=args)
    agents = []
    agent_labels = []
    avg_rewards = []
    avg_scores = []
    visit_counts = []
    win_rates = []
    win_statuses = []
    
    if args.monte:
        print("Training Monte Carlo agent...")
        monte_metrics = run_trials(MonteCarloAgent, args=args)
        agents.append(MonteCarloAgent)
        agent_labels.append("Monte Carlo")
        avg_rewards.append(monte_metrics["avg_rewards"])
        avg_scores.append(monte_metrics["avg_scores"])
        visit_counts.append(monte_metrics["state_action_visit_count"])
        win_rates.append(monte_metrics["avg_wins"])
        win_statuses.append(monte_metrics["win_statuses"])
  
    if args.sarsa:
        print("Training SARSA agent...")
        sarsa_metrics = run_trials(SARSA_0, args=args)
        agents.append(SARSA_0)
        agent_labels.append("SARSA")
        avg_rewards.append(sarsa_metrics["avg_rewards"])
        avg_scores.append(sarsa_metrics["avg_scores"])
        visit_counts.append(sarsa_metrics["state_action_visit_count"])
        win_rates.append(sarsa_metrics["avg_wins"])
        win_statuses.append(sarsa_metrics["win_statuses"])
    
    if args.qlearning:
        print("Training Q-Learning agent...")
        qlearning_metrics = run_trials(QLearningAgent, args=args)
        agents.append(QLearningAgent)
        agent_labels.append("Q-Learning")
        avg_rewards.append(qlearning_metrics["avg_rewards"])
        avg_scores.append(qlearning_metrics["avg_scores"])
        visit_counts.append(qlearning_metrics["state_action_visit_count"])
        win_rates.append(qlearning_metrics["avg_wins"])
        win_statuses.append(qlearning_metrics["win_statuses"])
        
    if args.monte_kate:
        print("Training Monte Carlo agent...")
        monte_metrics = run_trials(MonteCarlo, args=args)
        agents.append(MonteCarlo)
        agent_labels.append("Monte Carlo Kate")
        avg_rewards.append(monte_metrics["avg_rewards"])
        avg_scores.append(monte_metrics["avg_scores"])
        visit_counts.append(monte_metrics["state_action_visit_count"])
        win_rates.append(monte_metrics["avg_wins"])
        win_statuses.append(monte_metrics["win_statuses"])
  
    if args.sarsa_kate:
        print("Training SARSA agent...")
        sarsa_metrics = run_trials(SARSA, args=args)
        agents.append(SARSA)
        agent_labels.append("SARSA Kate")
        avg_rewards.append(sarsa_metrics["avg_rewards"])
        avg_scores.append(sarsa_metrics["avg_scores"])
        visit_counts.append(sarsa_metrics["state_action_visit_count"])
        win_rates.append(sarsa_metrics["avg_wins"])
        win_statuses.append(sarsa_metrics["win_statuses"])
    
    if args.qlearning_kate:
        print("Training Q-Learning agent...")
        qlearning_metrics = run_trials(QLearning, args=args)
        agents.append(QLearning)
        agent_labels.append("Q-Learning Kate")
        avg_rewards.append(qlearning_metrics["avg_rewards"])
        avg_scores.append(qlearning_metrics["avg_scores"])
        visit_counts.append(qlearning_metrics["state_action_visit_count"])
        win_rates.append(qlearning_metrics["avg_wins"])
        win_statuses.append(qlearning_metrics["win_statuses"])
        
    # Only plot if visualization is requested
    if args.plot:
        if args.monte:
            metrics.plot_agent_scores(agent_name="Monte Carlo", agent_scores=monte_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(monte_metrics["state_visit_percentages"], "Monte Carlo", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="Monte Carlo", visit_count=monte_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=monte_metrics["state_action_visit_count"], agent_name="Monte Carlo", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution(visit_count=monte_metrics["state_action_visit_count"], agent_name="Monte Carlo", save_path=METRICS_PATH)
        if args.sarsa:
            metrics.plot_agent_scores(agent_name="SARSA", agent_scores=sarsa_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(sarsa_metrics["state_visit_percentages"], "SARSA", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="SARSA", visit_count=sarsa_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA", save_path=METRICS_PATH)
        if args.qlearning:
            metrics.plot_agent_scores(agent_name="Q-Learning", agent_scores=qlearning_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(qlearning_metrics["state_visit_percentages"], "Q-Learning", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="Q-Learning", visit_count=qlearning_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning", save_path=METRICS_PATH)

        if args.monte_kate:
            metrics.plot_agent_scores(agent_name="Monte Carlo Kate", agent_scores=monte_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(monte_metrics["state_visit_percentages"], "Monte Carlo Kate", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="Monte Carlo Kate", visit_count=monte_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=monte_metrics["state_action_visit_count"], agent_name="Monte Carlo Kate", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution(visit_count=monte_metrics["state_action_visit_count"], agent_name="Monte Carlo Kate", save_path=METRICS_PATH)
        if args.sarsa_kate:
            metrics.plot_agent_scores(agent_name="SARSA Kate", agent_scores=sarsa_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(sarsa_metrics["state_visit_percentages"], "SARSA Kate", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="SARSA Kate", visit_count=sarsa_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA Kate", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA Kate", save_path=METRICS_PATH)
        if args.qlearning_kate:
            metrics.plot_agent_scores(agent_name="Q-Learning Kate", agent_scores=qlearning_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(qlearning_metrics["state_visit_percentages"], "Q-Learning Kate", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="Q-Learning Kate", visit_count=qlearning_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning Kate", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning Kate", save_path=METRICS_PATH)


        if len(agent_labels) > 1:
            metrics.plot_winning_percentage(agent_labels, win_rates, save_path=METRICS_PATH)
            metrics.plot_cumulative_return(avg_rewards, agent_labels, save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_percentage(visit_counts, agent_labels, save_path=METRICS_PATH)
            metrics.plot_all_agents_scores(avg_scores, agent_labels, save_path=METRICS_PATH)
            metrics.plot_winning_percentage_over_episodes(win_statuses, agent_labels, save_path=METRICS_PATH)
        else:
            print("At least two agents are required for comparison.")

        # Tune hyperparameters
        alpha_values = [0.01, 0.1, 0.5] 
        gamma_values = [0.5, 0.9, 0.95] 
        epsilon_values = [0.1, 0.2, 0.5] 

        # Run experiments for SARSA
        #run_trials_with_hyperparams(SARSA_0, alpha_values, gamma_values, epsilon_values, args)

        # Run experiments for Q-Learning
        #run_trials_with_hyperparams(QLearningAgent, alpha_values, gamma_values, epsilon_values, args)
