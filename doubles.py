import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Dict, Type, Union
import argparse

from QLearning_alayna import QLearningAgent
from SARSA_alayna import SARSA_0
from perfectAgent import PerfectAgent
from MonteCarlo_alayna import MonteCarloAgent
import metrics
from doublesPongEnv import PongEnv
from pongVisualizer import PongVisualizer
from MonteCarlo_kate import MonteCarlo
from SARSA_kate import SARSA
from QLearning_kate import QLearning

HERE = os.path.dirname(os.path.abspath(__file__))

AGENT_COUNT = 10
EPISODE_COUNT = 1000
WINDOW_LENGTH = 30
EXP_STARTS = False
DEBUG = True
METRICS_PATH = os.path.join(HERE, 'doubles-experiment1')

def log(val):
	if DEBUG:
		print(val)

if METRICS_PATH and not os.path.exists(METRICS_PATH):
        os.makedirs(METRICS_PATH)

def generate_episode(episode: int, env: PongEnv, agent_left: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], agent_right: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], visualizer=None) -> Tuple[List[float], np.ndarray, Tuple, bool]:
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
        state_index_left = env.get_state_index(agent="left")
        state_index_right = env.get_state_index(agent="right")

        # Each agent selects its action independently based on the state
        action_left = agent_left.select_action(state_index_left)
        action_right = agent_right.select_action(state_index_right)

        # Execute both actions in the environment
        new_state, reward, game_end = env.execute_action(action_left, action_right)
        next_state_index_left = env.get_state_index(agent="left")
        next_state_index_right = env.get_state_index(agent="right")
        
        log(f"Episode: {episode + 1}, New State: {new_state}, Reward: {reward}, Done: {game_end}")
        if DEBUG:
            env.render()
            
        reward_left, reward_right = reward
        rewards_left.append(reward_left)
        rewards_right.append(reward_right)
        
        # Track visits for each agent separately
        episode_visit_count_left[state_index_left, action_left] += 1
        episode_visit_count_right[state_index_right, action_right] += 1
        
        if game_end and reward_left > 0:
            win_left = True
        if game_end and reward_right > 0:
            win_right = True
        
        # Update both agents
        agent_left.update(next_state_index_left, reward_left)
        agent_right.update(next_state_index_right, -reward_right)  # If reward is positive for one, it's negative for the other
        
        if visualizer:
            ball_x, ball_y, paddle_y_left, paddle_y_right, ball_dx, ball_dy = env.get_state()
            visualizer.render((ball_x, ball_y), paddle_y_right)
        
        current_state = new_state
    
    # Handle final updates if agents need end-of-episode processing
    if isinstance(agent_left, (MonteCarlo, MonteCarloAgent)):
        agent_left.update_q()
        agent_left.clear_trajectory()
    if isinstance(agent_right, (MonteCarlo, MonteCarloAgent)):
        agent_right.update_q()
        agent_right.clear_trajectory()
  
    return rewards_left, rewards_right, episode_visit_count_left, episode_visit_count_right, current_state, win_left, win_right, env.get_score()


def run_trials(agent_left_class: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], agent_right_class: Type[Union[QLearningAgent, QLearning, SARSA_0, SARSA, MonteCarloAgent, MonteCarlo, PerfectAgent]], args):
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

    for i in range(AGENT_COUNT):
        # Alternate ball_dx direction for each environment instance
        initial_ball_dx = 1 if i % 2 == 0 else -1
        initial_ball_dy = np.random.choice([-1, 1])
        environment = PongEnv(grid_size=10, ball_dx=initial_ball_dx, ball_dy=initial_ball_dy)
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
        for episode in range(EPISODE_COUNT):
            log(f"Starting episode {episode + 1}")
            
            rewards_left, rewards_right, episode_visit_count_left, episode_visit_count_right, final_state, win_left, win_right, score = generate_episode(
                episode, environment, agent_left, agent_right, visualizer
            )
          
            score_left, score_right = score
            episode_rewards_left.append(sum(rewards_left))
            episode_rewards_right.append(sum(rewards_right))
            episode_scores_left.append(score_left)
            episode_scores_right.append(score_right)
            win_status_left.append(1 if win_left else 0)
            win_status_right.append(1 if win_right else 0)
            wins_left += win_left
            wins_right += win_right
            visit_count_left += episode_visit_count_left
            visit_count_right += episode_visit_count_right
            v_t_left = agent_left.get_visited_states_num()
            v_t_right = agent_right.get_visited_states_num()
            V_t_left[i,0] = (v_t_left/agent_left.get_number_of_states())*100
            V_t_right[i,0] = (v_t_right/agent_right.get_number_of_states())*100

            # Optionally, log more detailed information about the episode, such as win/loss
            log(f"Episode {episode + 1} finished. Left Agent Reward: {np.sum(rewards_left)}, Right Agent Reward: {np.sum(rewards_right)}")
            log(f"Final score: {score}, Win Left: {win_left}, Win Right: {win_right}")

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
    
    # Calculate average rewards over the last 30 episodes
    avg_rewards_last_30_left = np.mean([np.convolve(rewards, np.ones(30) / 30, mode='valid') for rewards in all_rewards_left], axis=0)

    # Calculate percentage of wins for the last 30 episodes
    recent_wins_left = np.mean([win_status[-30:] for win_status in all_wins_left], axis=0)
    avg_wins_last_30_left = np.mean(recent_wins_left)  

    # Visit count for states 0 to 450 only
    visit_count_0_to_450_left = visit_count_left[:450, :]
    percentage_visited_0_to_450_left = np.sum(visit_count_0_to_450_left > 0) / visit_count_0_to_450_left.size * 100  

    # Visit count for states 450 to 900
    visit_count_450_to_900_left = visit_count_left[450:9000, :]
    percentage_visited_450_to_900_left = np.sum(visit_count_450_to_900_left > 0) / visit_count_450_to_900_left.size * 100 

    print(str(agent_left_class.__name__))
    metrics.pretty_print_metrics(avg_rewards_last_30_left, avg_wins_last_30_left, percentage_visited_0_to_450_left, percentage_visited_450_to_900_left)

    # Calculate average rewards over the last 30 episodes
    avg_rewards_last_30_right = np.mean([np.convolve(rewards, np.ones(30) / 30, mode='valid') for rewards in all_rewards_right], axis=0)

    # Calculate percentage of wins for the last 30 episodes
    recent_wins_right = np.mean([win_status[-30:] for win_status in all_wins_right], axis=0)
    avg_wins_last_30_right = np.mean(recent_wins_right)  

    # Visit count for states 0 to 450 only
    visit_count_0_to_450_right = visit_count_right[:450, :]
    percentage_visited_0_to_450_right = np.sum(visit_count_0_to_450_right > 0) / visit_count_0_to_450_right.size * 100  

    # Visit count for states 450 to 900
    visit_count_450_to_900_right = visit_count_right[450:9000, :]
    percentage_visited_450_to_900_right = np.sum(visit_count_450_to_900_right > 0) / visit_count_450_to_900_right.size * 100 

    print(str(agent_right_class.__name__))
    metrics.pretty_print_metrics(avg_rewards_last_30_right, avg_wins_last_30_right, percentage_visited_0_to_450_right, percentage_visited_450_to_900_right)
    
    #return avg_reward_left, avg_reward_right, total_visits_left, total_visits_right
    return {
        'avg_rewards_left': np.mean(all_rewards_left, axis=0),
        'avg_wins_left': total_wins_left / (AGENT_COUNT * EPISODE_COUNT),  # Calculate win rate,
        'avg_scores_left': np.mean(all_scores_left, axis=0),
        'state_action_visit_count_left': visit_count_left,
        'win_statuses_left': np.mean(all_wins_left, axis=0),
        'state_visit_percentages_left': all_V_t_left,
        'avg_rewards_right': np.mean(all_rewards_right, axis=0),
        'avg_wins_right': total_wins_right / (AGENT_COUNT * EPISODE_COUNT),  # Calculate win rate,
        'avg_scores_right': np.mean(all_scores_right, axis=0),
        'state_action_visit_count_right': visit_count_right,
        'win_statuses_right': np.mean(all_wins_right, axis=0),
        'state_visit_percentages_right': all_V_t_right
    }

def run_trials_with_hyperparams(agent_left_class, agent_right_class, alpha_values: List[float], gamma_values: List[float], epsilon_values: List[float], args):
    # This will hold the results for each set of hyperparameters
    results = []

    best_avg_reward = -np.inf
    best_params = None
    
    trial_rewards_left = []
    trial_rewards_right = []
    trial_visits_left = np.zeros((8999, 3))
    trial_visits_right = np.zeros((8999, 3))

    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                print(f"Training {agent_left_class.__name__} vs {agent_right_class.__name__} with alpha={alpha}, gamma={gamma}, epsilon={epsilon}...")
                avg_reward_left, avg_reward_right, visits_left, visits_right = run_trials(agent_left, agent_right, args=args)

                # Store the results for this trial
                trial_rewards_left.append(avg_reward_left)
                trial_rewards_right.append(avg_reward_right)
                trial_visits_left += visits_left
                trial_visits_right += visits_right
                
                if avg_reward_left > best_avg_reward:
                    best_avg_reward = avg_reward_left
                    best_params = (alpha, gamma, epsilon)

    if best_params:
        print("\n" + "*" * 50)
        print(f"***** Best Parameters Found *****")
        print(f"* Alpha:    {best_params[0]:.4f}")
        print(f"* Gamma:    {best_params[1]:.4f}")
        print(f"* Epsilon:  {best_params[2]:.4f}")
        print(f"* Avg Reward: {best_avg_reward:.2f}")
        print("*" * 50 + "\n")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_left', type=str, required=True, help="The type of the left agent (e.g., 'AgentType1')")
    parser.add_argument('--agent_right', type=str, required=True, help="The type of the right agent (e.g., 'AgentType2')")
    parser.add_argument('--sarsa', action='store_true', help='if SARSA algorithm should be run')
    parser.add_argument('--monte', action='store_true', help='if Monte Carlo algorithm should be run')
    parser.add_argument('--qlearning', action='store_true', help='if Q-Learning algorithm should be run')
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
    else:
        raise ValueError(f"Unknown agent type for left agent: {args.agent_left}")

    if args.agent_right.lower() == 'monte':
        agent_right = MonteCarloAgent 
    elif args.agent_right == 'sarsa':
        agent_right = SARSA_0
    elif args.agent_right == 'qlearning':
        agent_right = QLearningAgent
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
    agents.append(agent_left)
    agent_labels.append(str(agent_left.__name__))
    avg_rewards.append(results["avg_rewards_left"])
    avg_scores.append(results["avg_scores_left"])
    visit_counts.append(results["state_action_visit_count_left"])
    win_rates.append(results["avg_wins_left"])
    win_statuses.append(results["win_statuses_left"])
    
    agents.append(agent_right)
    agent_labels.append(str(agent_right.__name__))
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

