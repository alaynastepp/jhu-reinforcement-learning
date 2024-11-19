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
from doublesPongEnv_testing import PongEnv
from pongVisualizer import PongVisualizer
from kate_files.MonteCarlo_agent import MonteCarlo
from kate_files.SARSA_agent import SARSA
from kate_files.QLearning_agent import QLearning

HERE = os.path.dirname(os.path.abspath(__file__))

EPISODE_COUNT = 1000
WINDOW_LENGTH = 30
EXP_STARTS = False
DEBUG = False
METRICS_PATH = os.path.join(HERE, 'doubles-experiment1')
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

def generate_episode(episode: int, env: PongEnv, agent_left: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], agent_right: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], visualizer=None) -> Tuple[List[float], np.ndarray, Tuple, bool]:
    """
    Play one episode in the Pong environment with two agents (left and right) and collect rewards.

    :param episode (int): Current episode number.
    :param env (PongEnv): The Pong environment.
    :param agent_left: The agent controlling the left paddle.
    :param agent_right: The agent controlling the right paddle.
    :param visualizer: Optional visualizer to render each step.

    ::return (dict): A dictionary containing the following keys:
        - 'rewards_left' (List[float]): Rewards collected by the left agent during the episode.
        - 'rewards_right' (List[float]): Rewards collected by the right agent during the episode.
        - 'episode_visit_count_left' (np.ndarray): State-action visit counts for the left agent.
        - 'episode_visit_count_right' (np.ndarray): State-action visit counts for the right agent.
        - 'current_state' (Tuple): The final state of the environment after the episode ends.
        - 'win_left' (bool): True if the left agent wins the episode, False otherwise.
        - 'win_right' (bool): True if the right agent wins the episode, False otherwise.
        - 'env_score' (int): The final score of the environment after the episode ends.
    """
    current_state = env.reset()
    log(f"Initial ball position: {env.ball_x}, paddle positions: {env.paddle_y_left}, {env.paddle_y_right}")
    
    game_end = False
    rewards_left = []
    rewards_right = []
    win_left = False
    win_right = False
    episode_visit_count_left = np.zeros((env.get_number_of_states(), env.get_number_of_actions()))
    episode_visit_count_right = np.zeros((env.get_number_of_states(), env.get_number_of_actions()))
    
    while not game_end:
        state_index_left = env.get_state_index()
        state_index_right = env.get_state_index()

        # Each agent selects its action independently based on the state
        action_left = agent_left.select_action(state_index_left)
        action_right = agent_right.select_action(state_index_right)

        # Execute both actions in the environment
        new_state, reward, game_end = env.execute_action(action_left, action_right)
        next_state_index_left = env.get_state_index()
        next_state_index_right = env.get_state_index()
        
        log(f"Episode: {episode + 1}, New State: {new_state}, Reward: {reward}, Done: {game_end}")
        if DEBUG:
            env.render()
            
        reward_left, reward_right = reward
        rewards_left.append(reward_left)
        rewards_right.append(reward_right)
        
        # Track visits for each agent separately
        episode_visit_count_left[state_index_left, action_left] += 1
        episode_visit_count_right[state_index_right, action_right] += 1
        
        if game_end and (sum(rewards_left) > 0):
            win_left = True
        if game_end and (sum(rewards_right) > 0):
            win_right = True
        
        # Update both agents
        agent_left.update(next_state_index_left, reward_left)
        agent_right.update(next_state_index_right, -reward_right)  # If reward is positive for one, it's negative for the other
        
        if visualizer:
            ball_x, ball_y, paddle_y_left, paddle_y_right, ball_dx, ball_dy = env.get_state()
            visualizer.render_dynamic((ball_x, ball_y), paddle_y_left, paddle_y_right)
        
        current_state = new_state
    
    # Handle final updates if agents need end-of-episode processing
    if isinstance(agent_left, (MonteCarlo, MonteCarloAgent)):
        agent_left.update_q()
        agent_left.clear_trajectory()
    if isinstance(agent_right, (MonteCarlo, MonteCarloAgent)):
        agent_right.update_q()
        agent_right.clear_trajectory()
  
    return {
        'rewards_left': rewards_left,
        'rewards_right': rewards_right,
        'episode_visit_count_left': episode_visit_count_left,
        'episode_visit_count_right': episode_visit_count_right,
        'current_state': current_state,
        'win_left': win_left,
        'win_right': win_right,
        'score': env.get_score()
    }


def run_trials(agent_left_class: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], agent_right_class: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], args):
    """
	Based on the agent type passed in, run many agents for a certain amount of episodes and gather metrics on their performance

	:param agent_class (class): One of the following: SARSA_0, QLearningAgent, or MonteCarlo.
    :param args (argparse.Namespace): Parsed arguments from argparse containing parameters such as alpha, gamma, and epsilon. Can also contain 'pretrained' if we want to use a pretrained agent.
    :return Dict containing the following metrics for each agent (left and right):
            'avg_rewards': np.ndarray - Average rewards over all agents.
            'avg_wins': float - Overall win rate.
            'avg_scores': np.ndarray - Average scores over all agents.
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
    print(f"Running trials for {agent_left_class.__name__} vs {agent_right_class.__name__} with non-default args {params}")

    all_rewards_left = []
    all_scores_left = []
    all_wins_left = []
    total_wins_left = 0
    visit_count_left = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions())) #raw count of how many times a specific state-action pair has been visited across episodes
    all_V_t_left = []
    all_rewards_right = []
    all_scores_right = []
    all_wins_right = []
    total_wins_right = 0
    visit_count_right = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions())) #raw count of how many times a specific state-action pair has been visited across episodes
    all_V_t_right = []

    environment = PongEnv(grid_size=10)
    if args.pretrained:
        agent_left = load_agent(agent_left_class, os.path.join(TRAINED_AGENTS_PATH, f'trained_{str(agent_left_class.__name__)}_9.pkl'), environment.get_number_of_states(), environment.get_number_of_actions()) #, gamma=0.9, learning_rate=0.1, epsilon=0.1)
        agent_right = load_agent(agent_right_class, os.path.join(TRAINED_AGENTS_PATH, f'trained_{str(agent_right_class.__name__)}_9.pkl'), environment.get_number_of_states(), environment.get_number_of_actions()) #, gamma=0.9, learning_rate=0.1, epsilon=0.1)
    else:
        agent_left = agent_left_class(environment.get_number_of_states(), environment.get_number_of_actions(), **params)
        agent_right = agent_right_class(environment.get_number_of_states(), environment.get_number_of_actions(), **params)
    
    episode_rewards_left = []
    episode_scores_left = []
    win_status_left = []
    V_t_left = np.zeros((EPISODE_COUNT,1))  # percent states visited per episode
    wins_left = 0
    episode_rewards_right = []
    episode_scores_right = []
    win_status_right = []
    V_t_right = np.zeros((EPISODE_COUNT,1))  # percent states visited per episode
    wins_right = 0
    for i in range(EPISODE_COUNT):
        # Alternate ball_dx direction for each episode 
        initial_ball_dx = 1 if i % 2 == 0 else -1
        initial_ball_dy = 1 if (i // 2) % 2 == 0 else -1
        environment = PongEnv(grid_size=10, ball_dx=initial_ball_dx, ball_dy=initial_ball_dy)
    
        log(f"Starting episode {i + 1}")
        
        results = generate_episode(i, environment, agent_left, agent_right, visualizer)
      
        score_left, score_right = results['score']
        episode_rewards_left.append(sum(results['rewards_left']))
        episode_rewards_right.append(sum(results['rewards_right']))
        episode_scores_left.append(score_left)
        episode_scores_right.append(score_right)
        win_status_left.append(1 if results['win_left'] else 0)
        win_status_right.append(1 if results['win_right'] else 0)
        wins_left += results['win_left']
        wins_right += results['win_right']
        visit_count_left += results['episode_visit_count_left']
        visit_count_right += results['episode_visit_count_right']
        v_t_left = agent_left.get_visited_states_num()
        v_t_right = agent_right.get_visited_states_num()
        V_t_left[i,0] = (v_t_left/agent_left.get_number_of_states())*100
        V_t_right[i,0] = (v_t_right/agent_right.get_number_of_states())*100
        # Optionally, log more detailed information about the episode, such as win/loss
        log(f"Episode {i + 1} finished. Left Agent Reward: {np.sum(results['rewards_left'])}, Right Agent Reward: {np.sum(results['rewards_right'])}")
        log(f"Final score: {results['score']}, Win Left: {results['win_left']}, Win Right: {results['win_right']}")
    all_rewards_left.append(episode_rewards_left)
    all_rewards_right.append(episode_rewards_right)
    all_scores_left.append(episode_scores_left)
    all_scores_right.append(episode_scores_right)
    total_wins_left += wins_left
    total_wins_right += wins_right
    all_wins_left.append(win_status_left)
    all_wins_right.append(win_status_right)
    all_V_t_left.append(V_t_left.flatten())
    all_V_t_right.append(V_t_right.flatten())
    
    # Calculate average rewards over all episodes
    avg_rewards_all_left = np.mean([np.mean(rewards) for rewards in all_rewards_left])
    
    # Calculate percentage of wins for all episodes
    avg_wins_all_left = np.mean([np.mean(win_status) for win_status in all_wins_left])

    # Visit count for states 0 to 450 only
    visit_count_0_to_450_left = visit_count_left[:450, :]
    percentage_visited_0_to_450_left = np.sum(visit_count_0_to_450_left > 0) / visit_count_0_to_450_left.size * 100  

    # Visit count for states 450 to 900
    visit_count_450_to_900_left = visit_count_left[450:9000, :]
    percentage_visited_450_to_900_left = np.sum(visit_count_450_to_900_left > 0) / visit_count_450_to_900_left.size * 100 

    print("\nLeft agent:", str(agent_left_class.__name__))
    metrics.pretty_print_metrics_all_ep(avg_rewards_all_left, avg_wins_all_left, percentage_visited_0_to_450_left, percentage_visited_450_to_900_left)

    # Calculate average rewards over all episodes
    avg_rewards_all_right = np.mean([np.mean(rewards) for rewards in all_rewards_right])

    # Calculate percentage of wins for all episodes
    avg_wins_all_right = np.mean([np.mean(win_status) for win_status in all_wins_right])

    # Visit count for states 0 to 450 only
    visit_count_0_to_450_right = visit_count_right[:450, :]
    percentage_visited_0_to_450_right = np.sum(visit_count_0_to_450_right > 0) / visit_count_0_to_450_right.size * 100  

    # Visit count for states 450 to 900
    visit_count_450_to_900_right = visit_count_right[450:9000, :]
    percentage_visited_450_to_900_right = np.sum(visit_count_450_to_900_right > 0) / visit_count_450_to_900_right.size * 100 

    # Print results
    print("Right Agent:", str(agent_right_class.__name__))
    metrics.pretty_print_metrics_all_ep(avg_rewards_all_right, avg_wins_all_right, percentage_visited_0_to_450_right, percentage_visited_450_to_900_right)

    #return avg_reward_left, avg_reward_right, total_visits_left, total_visits_right
    return {
        'avg_rewards_left': np.mean(all_rewards_left, axis=0),
        'avg_wins_left': total_wins_left / EPISODE_COUNT,  # Calculate win rate,
        'avg_scores_left': np.mean(all_scores_left, axis=0),
        'state_action_visit_count_left': visit_count_left,
        'win_statuses_left': np.mean(all_wins_left, axis=0),
        'state_visit_percentages_left': all_V_t_left,
        'avg_rewards_right': np.mean(all_rewards_right, axis=0),
        'avg_wins_right': total_wins_right / EPISODE_COUNT,  # Calculate win rate,
        'avg_scores_right': np.mean(all_scores_right, axis=0),
        'state_action_visit_count_right': visit_count_right,
        'win_statuses_right': np.mean(all_wins_right, axis=0),
        'state_visit_percentages_right': all_V_t_right
    }

def run_trials_with_hyperparams(agent_left_class, agent_right_class, alpha_values: List[float], gamma_values: List[float], epsilon_values: List[float], args):
    """
    Runs multiple trials with different hyperparameter values and identifies the best configuration.

    :param agent_class: The agent class to use for training.
    :param alpha_values: A list of possible alpha (learning rate) values.
    :param gamma_values: A list of possible gamma (discount factor) values.
    :param epsilon_values: A list of possible epsilon (exploration rate) values.
    """
    best_avg_reward_left = -np.inf
    best_params_left = None
    
    best_avg_reward_right = -np.inf
    best_params_right = None

    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                print(f"Training {agent_left_class.__name__} vs {agent_right_class.__name__} with alpha={alpha}, gamma={gamma}, epsilon={epsilon}...")
                metrics = run_trials(agent_left, agent_right, args=args)
                
                avg_reward_left = np.mean(metrics['avg_rewards_left'])
                if avg_reward_left > best_avg_reward_left:
                    best_avg_reward_left = avg_reward_left
                    best_params_left = (alpha, gamma, epsilon)
                    
                avg_reward_right = np.mean(metrics['avg_rewards_right'])
                if avg_reward_right > best_avg_reward_right:
                    best_avg_reward_right = avg_reward_right
                    best_params_right = (alpha, gamma, epsilon)

    if best_params_left:
        print("\n" + "*" * 50)
        print(f"***** Best Parameters Found for Left Agent {agent_left_class.__name__} *****")
        print(f"* Alpha:    {best_params_left[0]:.4f}")
        print(f"* Gamma:    {best_params_left[1]:.4f}")
        print(f"* Epsilon:  {best_params_left[2]:.4f}")
        print(f"* Avg Reward: {best_params_left:.2f}")
        print("*" * 50 + "\n")
    
    if best_params_right:
        print("\n" + "*" * 50)
        print(f"***** Best Parameters Found for Right Agent {agent_right_class.__name__} *****")
        print(f"* Alpha:    {best_params_right[0]:.4f}")
        print(f"* Gamma:    {best_params_right[1]:.4f}")
        print(f"* Epsilon:  {best_params_right[2]:.4f}")
        print(f"* Avg Reward: {best_params_right:.2f}")
        print("*" * 50 + "\n")

def save_agent(agent, filename):
    """
    Save the Q-table of an agent to a file using pickle.

    :param agent: The agent whose Q-table is to be saved.
    :param path (str): The file path where the Q-table will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print(f"Agent saved to {filename}")

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
    parser.add_argument('--agent_left', type=str, required=True, help="The type of the left agent (e.g., 'AgentType1')")
    parser.add_argument('--agent_right', type=str, required=True, help="The type of the right agent (e.g., 'AgentType2')")
    parser.add_argument('--pretrained', action='store_true', help='if we want to load a pre-trained agent')
    parser.add_argument('--viz', action='store_true', help="if visualization is wanted")
    parser.add_argument('--plot', action='store_true', help="if plotting is wanted")
    parser.add_argument('--gamma', help="the value to be used for gamma")
    parser.add_argument('--learningrate', help='the value to be used for learning rate')
    parser.add_argument('--epsilon', help='the value to be used for epsilon')
    parser.add_argument('--debug', help='if debug mode is turned on')
    
    args = parser.parse_args()
    
    if args.agent_left.lower() == 'monte':
        agent_left = MonteCarloAgent
    elif args.agent_left == 'sarsa':
        agent_left = SARSA_0
    elif args.agent_left == 'qlearning':
        agent_left = QLearningAgent
    elif args.agent_left.lower() == 'monte_kate':
        agent_left = MonteCarlo
    elif args.agent_left == 'sarsa_kate':
        agent_left = SARSA
    elif args.agent_left == 'qlearning_kate':
        agent_left = QLearning
    else:
        raise ValueError(f"Unknown agent type for left agent: {args.agent_left}")

    if args.agent_right.lower() == 'monte':
        agent_right = MonteCarloAgent 
    elif args.agent_right == 'sarsa':
        agent_right = SARSA_0
    elif args.agent_right == 'qlearning':
        agent_right = QLearningAgent
    elif args.agent_right.lower() == 'monte_kate':
        agent_right = MonteCarlo
    elif args.agent_right == 'sarsa_kate':
        agent_right = SARSA
    elif args.agent_right == 'qlearning_kate':
        agent_right = QLearning
    else:
        raise ValueError(f"Unknown agent type for left agent: {args.agent_right}")
    
    agents = []
    agent_labels = []
    avg_rewards = []
    avg_scores = []
    visit_counts = []
    win_rates = []
    win_statuses = []
    
    results = run_trials(agent_left, agent_right, args)
    
    if agent_left.__name__ == agent_right.__name__:
        agent_labels.append(f"{str(agent_left.__name__)}_left")
        agent_labels.append(f"{str(agent_right.__name__)}_right")
    else:
        agent_labels.append(str(agent_left.__name__))
        agent_labels.append(str(agent_right.__name__))
    
    agents.append(agent_left)
    avg_rewards.append(results["avg_rewards_left"])
    avg_scores.append(results["avg_scores_left"])
    visit_counts.append(results["state_action_visit_count_left"])
    win_rates.append(results["avg_wins_left"])
    win_statuses.append(results["win_statuses_left"])
    
    agents.append(agent_right)
    avg_rewards.append(results["avg_rewards_right"])
    avg_scores.append(results["avg_scores_right"])
    visit_counts.append(results["state_action_visit_count_right"])
    win_rates.append(results["avg_wins_right"])
    win_statuses.append(results["win_statuses_right"])

    # Only plot if visualization is requested
    if args.plot:
        metrics.plot_agent_scores(agent_name=str(agent_left.__name__), agent_scores=results["avg_scores_left"], save_path=METRICS_PATH)
        metrics.plot_state_visitation(results["state_visit_percentages_left"], str(agent_left.__name__), save_path=METRICS_PATH)
        metrics.plot_visit_percentage(agent_name=str(agent_left.__name__), visit_count=results["state_action_visit_count_left"], save_path=METRICS_PATH)
        metrics.plot_mean_visited_states_per_action(visit_count=results["state_action_visit_count_left"], agent_name=str(agent_left.__name__), save_path=METRICS_PATH)
        metrics.plot_state_action_distribution(visit_count=results["state_action_visit_count_left"], agent_name=str(agent_left.__name__), save_path=METRICS_PATH)
        metrics.plot_agent_scores(agent_name=str(agent_right.__name__), agent_scores=results["avg_scores_right"], save_path=METRICS_PATH)
        metrics.plot_state_visitation(results["state_visit_percentages_right"], str(agent_right.__name__), save_path=METRICS_PATH)
        metrics.plot_visit_percentage(agent_name=str(agent_right.__name__), visit_count=results["state_action_visit_count_right"], save_path=METRICS_PATH)
        metrics.plot_mean_visited_states_per_action(visit_count=results["state_action_visit_count_right"], agent_name=str(agent_right.__name__), save_path=METRICS_PATH)
        metrics.plot_state_action_distribution(visit_count=results["state_action_visit_count_right"], agent_name=str(agent_right.__name__), save_path=METRICS_PATH)

        if len(agent_labels) > 1:
            metrics.plot_winning_percentage(agent_labels, win_rates, save_path=METRICS_PATH)
            metrics.plot_cumulative_return(avg_rewards, agent_labels, save_path=METRICS_PATH)
            metrics.plot_mean_visited_states_percentage(visit_counts, agent_labels, save_path=METRICS_PATH)
            metrics.plot_all_agents_scores(avg_scores, agent_labels, save_path=METRICS_PATH)
            metrics.plot_winning_percentage_over_episodes(win_statuses, agent_labels, save_path=METRICS_PATH)
        else:
            print("At least two agents are required for comparison.")

    
    alpha_values = [0.01, 0.1, 0.5]  # Example learning rates
    gamma_values = [0.5, 0.9, 0.95]  # Example discount factors
    epsilon_values = [0.1, 0.2, 0.5]  # Example exploration rates
    # run_trials_with_hyperparams(agent_left, agent_right, alpha_values, gamma_values, epsilon_values, args=args)

