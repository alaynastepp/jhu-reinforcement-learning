import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from typing import List, Tuple, Dict, Type, Union
import argparse
import pickle

from agents.QLearning import QLearning
from agents.SARSA import SARSA
from agents.MonteCarlo import MonteCarlo
from alayna_agents.perfect_agent import PerfectAgent
import metrics
from pongEnv import PongEnv
from pongVisualizer import PongVisualizer


HERE = os.path.dirname(os.path.abspath(__file__))

AGENT_COUNT = 10
EPISODE_COUNT = 1000
DEBUG = False
METRICS_PATH = os.path.join(HERE, 'experiment1')
TRAINED_AGENTS_PATH = os.path.join(HERE, 'best_agents')

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
        
def generate_episode(episode: int, env: PongEnv, agent: Type[Union[QLearning, SARSA, MonteCarlo, PerfectAgent]], visualizer=None) -> Tuple[List[float], np.ndarray, Tuple, bool]:
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
        
    if type(agent) is MonteCarlo:
        agent.update_q()
        agent.clear_trajectory()
  
    # return the result of the game
    return rewards, episode_visit_count, current_state, win, env.get_score()

def reset_environment(args):
    if args.left:
        return PongEnv(grid_size=10, agent_side="left")
    elif args.right:
        return PongEnv(grid_size=10, agent_side="right")
    else:
        return PongEnv(grid_size=10)

def run_trials(agent_class: Type[Union[QLearning, SARSA, MonteCarlo, PerfectAgent]], args : argparse.Namespace) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_class (class): One of the following: SARSA, QLearning, or MonteCarlo.
    :param args (argparse.Namespace): Parsed arguments from argparse containing parameters such as alpha, gamma, and epsilon.
    :return Dict containing the following metrics:
            'avg_rewards': np.ndarray - Average rewards over all agents.
            'avg_wins': float - Overall win rate.
            'avg_scores': np.ndarray - Average scores over all agents.
            'state_action_visit_count': np.ndarray - Visit counts for each state-action pair.
            'win_statuses': np.ndarray - Array of win status for each episode.
            'state_visit_percentages': List[float] - State visit percentages across episodes.
	"""
    
    environment = reset_environment(args)

    if args.viz:
        visualizer = PongVisualizer(grid_size=10, cell_size=60)
    else:
        visualizer = None
        
    params = {"gamma": args.gamma, "alpha": args.alpha, "epsilon": args.epsilon}
    params = {k:float(v) for k,v in params.items() if v is not None}
    print(f"Running trials for {agent_class} with non-default args {params}")

    all_rewards = []
    all_scores = []
    all_wins = []
    total_wins = 0
    visit_count = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions())) #raw count of how many times a specific state-action pair has been visited across episodes
    all_V_t = [] #percentage of the total states that have been visited 
    
    for a in range(AGENT_COUNT):
        environment = reset_environment(args)
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
            save_agent(agent, os.path.join(TRAINED_AGENTS_PATH, f'{environment.agent_side}_trained_{str(agent_class.__name__)}_a{agent.alpha}_g{agent.gamma}_e{agent.epsilon}_{a}.pkl'))
        
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
    
    return {
        'avg_rewards': np.mean(all_rewards, axis=0),
        'avg_wins': total_wins / (AGENT_COUNT * EPISODE_COUNT),  # Calculate win rate
        'avg_scores': np.mean(all_scores, axis=0),
        'state_action_visit_count': visit_count / AGENT_COUNT,
        'win_statuses': np.mean(all_wins, axis=0),
        'state_visit_percentages': all_V_t
    }

def run_trials_with_hyperparams(agent_class: Type[Union[QLearning, SARSA, MonteCarlo, PerfectAgent]], alpha_values: List[float], gamma_values: List[float], epsilon_values: List[float], args) -> None:
    """
    Runs multiple trials with different hyperparameter values and identifies the best configuration.

    :param agent_class: The agent class to use for training.
    :param alpha_values: A list of possible alpha (learning rate) values.
    :param gamma_values: A list of possible gamma (discount factor) values.
    :param epsilon_values: A list of possible epsilon (exploration rate) values.
    """
    best_avg_reward = -np.inf
    best_params = []

    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                for a in range(5):
                    print(f"Training {agent_class.__name__} with alpha={alpha}, gamma={gamma}, epsilon={epsilon}...")

                    args.alpha = alpha
                    args.gamma = gamma
                    args.epsilon = epsilon
                    
                    metrics = run_trials(agent_class, args=args)
                    
                    avg_reward = np.mean(metrics['avg_rewards'])

                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_params = [(alpha, gamma, epsilon)]
                    elif avg_reward == best_avg_reward:
                        best_params.append((alpha, gamma, epsilon))

    for params in best_params:
        print("\n" + "*" * 50)
        print(f"***** Best Parameters Found *****")
        print(f"* Alpha:    {params[0]:.4f}")
        print(f"* Gamma:    {params[1]:.4f}")
        print(f"* Epsilon:  {params[2]:.4f}")
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


def createDict(label, agent, metrics):
    return {
        "label": label,
        "agent": agent,
        "rewards": metrics['avg_rewards'],
        "scores": metrics['avg_scores'],
        "visits": metrics['state_action_visit_count'],
        "win_rates": metrics['avg_wins'],
        "win_statuses": metrics['win_statuses'],
        "visit_percentages": metrics["state_visit_percentages"]
    }


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sarsa', action='store_true', help='if SARSA algorithm should be run')
    parser.add_argument('--monte', action='store_true', help='if Monte Carlo algorithm should be run')
    parser.add_argument('--qlearning', action='store_true', help='if Q-Learning algorithm should be run')
    parser.add_argument('--viz', action='store_true', help="if visualization is wanted")
    parser.add_argument('--plot', action='store_true', help="if plotting is wanted")
    parser.add_argument('--gamma', help="the value to be used for discount factor")
    parser.add_argument('--alpha', help='the value to be used for learning rate')
    parser.add_argument('--epsilon', help='the value to be used for exploration rate')
    parser.add_argument('--left', action='store_true', help='if the agent is on the left side of the baord')
    parser.add_argument('--right', action='store_true', help='if the agent is on the right side of the board')
    parser.add_argument('--save', action='store_true', help='if the agent should get saved off as a pretrained agent after training')
    
    args = parser.parse_args()

    results = []
    
    if args.monte:
        print("Training Monte Carlo agent...")
        monte_metrics = run_trials(MonteCarlo, args=args)
        results.append(createDict("Monte Carlo", MonteCarlo, monte_metrics))
  
    if args.sarsa:
        print("Training SARSA agent...")
        sarsa_metrics = run_trials(SARSA, args=args)
        results.append(createDict("SARSA", SARSA, sarsa_metrics))
    
    if args.qlearning:
        print("Training Q-Learning agent...")
        qlearning_metrics = run_trials(QLearning, args=args)
        results.append(createDict("Q-Learning", QLearning, qlearning_metrics))
        
    # Only plot if visualization is requested
    if args.plot:
        if args.monte:
            metrics.plot_agent_scores(agent_name="Monte Carlo", agent_scores=monte_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(monte_metrics["state_visit_percentages"], "Monte Carlo", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="Monte Carlo", visit_count=monte_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=monte_metrics["state_action_visit_count"], agent_name="Monte Carlo", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution_logscale(visit_count=monte_metrics["state_action_visit_count"], agent_name="Monte Carlo", save_path=METRICS_PATH)
        if args.sarsa:
            metrics.plot_agent_scores(agent_name="SARSA", agent_scores=sarsa_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(sarsa_metrics["state_visit_percentages"], "SARSA", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="SARSA", visit_count=sarsa_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution_logscale(visit_count=sarsa_metrics["state_action_visit_count"], agent_name="SARSA", save_path=METRICS_PATH)
        if args.qlearning:
            metrics.plot_agent_scores(agent_name="Q-Learning", agent_scores=qlearning_metrics["avg_scores"], save_path=METRICS_PATH)
            metrics.plot_state_visitation(qlearning_metrics["state_visit_percentages"], "Q-Learning", save_path=METRICS_PATH)
            metrics.plot_visit_percentage(agent_name="Q-Learning", visit_count=qlearning_metrics["state_action_visit_count"], save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_per_action(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning", save_path=METRICS_PATH)
            metrics.plot_state_action_distribution_logscale(visit_count=qlearning_metrics["state_action_visit_count"], agent_name="Q-Learning", save_path=METRICS_PATH)

        if len(results) > 1:
            labels = [x['label'] for x in results]
            metrics.plot_winning_percentage(labels, [x['win_rates'] for x in results], save_path=METRICS_PATH)
            metrics.plot_cumulative_return([x['rewards'] for x in results], labels, save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_percentage([x['visits'] for x in results], labels, save_path=METRICS_PATH)
            metrics.plot_all_agents_scores([x['scores'] for x in results], labels, save_path=METRICS_PATH)
            metrics.plot_all_agents_scores_smoothed([x['scores'] for x in results], labels, save_path=METRICS_PATH)
            metrics.plot_winning_percentage_over_episodes([x['win_statuses'] for x in results], labels, save_path=METRICS_PATH)
            metrics.plot_state_action_distribution_all_logscale([x['visits'] for x in results], labels, save_path=METRICS_PATH)
            metrics.plot_state_visitation_all([x['visit_percentages'] for x in results], labels, save_path=METRICS_PATH)
        else:
            print("At least two agents are required for comparison.")

    # Tune hyperparameters
    alpha_values = [0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.8] 
    gamma_values = [0.1, 0.5, 0.9, 0.95] 
    epsilon_values = [0.01, 0.1, 0.2, 0.5] 

    # Run experiments for SARSA
    #run_trials_with_hyperparams(SARSA, alpha_values, gamma_values, epsilon_values, args)

    # Run experiments for Q-Learning
    #run_trials_with_hyperparams(QLearning, alpha_values, gamma_values, epsilon_values, args)

    # Run experiments for Monte Carlo
    #run_trials_with_hyperparams(MonteCarlo, alpha_values, gamma_values, epsilon_values, args)