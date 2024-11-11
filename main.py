import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from basicPongEnv import PongEnv
#from exampleAgent import Agent
from alaynaEnv import PongEnv
from QLearning_agent import QLearingAgent
from SARSA_agent import SARSA_0
from perfectAgent import PerfectAgent
from pongVisualizer import PongVisualizer
import metrics

AGENT_COUNT = 10
EPISODE_COUNT = 1000
WINDOW_LENGTH = 30
EXP_STARTS = False

def generate_episode(episode, env, agent, visualizer=None, debug=False):
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
    if debug:
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
        if debug:
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
    #visualizer = PongVisualizer(grid_size=10, cell_size=60)
    # TODO: establish metrics for each agent
    all_rewards = []
    all_wins = []
    total_wins = 0
    visit_count = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions()))
    for i in range(AGENT_COUNT):
        if agent_class == PerfectAgent:
            agent = agent_class(environment) 
        else:
            agent = agent_class(environment.get_number_of_states(), environment.get_number_of_actions())
        # TODO: initialize arrays for keeping track of agent performance over time
        episode_rewards = []
        win_status = []
        V_t = np.zeros((EPISODE_COUNT,1))  # percent states visited per episod
        wins = 0
        for i in range(EPISODE_COUNT): 
            # play game
            rewards, episode_visit_count, final_state, win = generate_episode(i, environment, agent, debug=True) #, visualizer=visualizer
            episode_rewards.append(sum(rewards))
            win_status.append(1 if win else 0)
            wins += win
            # TODO: record metrics
            if agent_class != PerfectAgent:
                v_t = agent.get_state_actn_visits()
                V_t[i,0] = (v_t/agent.get_number_of_states())*100
            #agent.clear_trajectory()
        print("EPISODE REWARDS ", episode_rewards)
        # TODO: return arrays full of metrics averaged over all agents
        all_rewards.append(episode_rewards)
        total_wins += wins
        all_wins.append(win_status)
        visit_count += episode_visit_count
        #visualizer.close()
        
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

    metrics.pretty_print_metrics(avg_rewards_last_30, avg_wins_last_30, percentage_visited_0_to_63, percentage_visited_64_to_127)

    return avg_rewards, avg_wins, visit_count, np.mean(all_wins, axis=0)

def run_trials_with_hyperparams(agent_class, alpha_values, gamma_values, epsilon_values):

    environment = PongEnv(grid_size=10)
    #visualizer = PongVisualizer(grid_size=10, cell_size=60)
    best_avg_reward = -np.inf
    best_params = None
    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon in epsilon_values:
                print(f"Training {agent_class.__name__} with alpha={alpha}, gamma={gamma}, epsilon={epsilon}...")
                # TODO: establish metrics for each agent
                total_rewards = []
                all_rewards = []
                all_wins = []
                total_wins = 0
                visit_count = np.zeros((environment.get_number_of_states(), environment.get_number_of_actions()))
                for i in range(AGENT_COUNT):
                    agent = agent_class(environment.get_number_of_states(), environment.get_number_of_actions())
                    # TODO: initialize arrays for keeping track of agent performance over time
                    episode_rewards = []
                    win_status = []
                    V_t = np.zeros((EPISODE_COUNT,1))  # percent states visited per episod
                    wins = 0
                    for i in range(EPISODE_COUNT): 
                        # play game
                        rewards, episode_visit_count, final_state, win = generate_episode(i, environment, agent) #, visualizer=visualizer
                        episode_rewards.append(sum(rewards))
                        win_status.append(1 if win else 0)
                        wins += win
                        # TODO: record metrics
                        v_t = agent.get_state_actn_visits()
                        V_t[i,0] = (v_t/agent.get_number_of_states())*100
                        #agent.clear_trajectory()
                    # TODO: return arrays full of metrics averaged over all agents
                    total_rewards.append(np.mean(episode_rewards))
                    all_rewards.append(episode_rewards)
                    total_wins += wins
                    all_wins.append(win_status)
                    visit_count += episode_visit_count
                    #visualizer.close()
                
                avg_reward = np.mean(total_rewards)
                avg_win_rate = total_wins / (AGENT_COUNT * EPISODE_COUNT)
                
                # Calculate average reward over the last 30 episodes for each agent
                avg_reward_last_30 = np.mean([np.mean(agent_rewards) for agent_rewards in all_rewards])

                # Calculate average win rate over the last 30 episodes
                recent_wins = np.mean([win_status[-30:] for win_status in all_wins], axis=0)
                avg_wins_last_30 = np.mean(recent_wins)  

                # Update the best parameters if current average reward is higher
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_params = (alpha, gamma, epsilon)

                print(f"Average Rewards (last 30 episodes): {avg_reward_last_30:.2f}, Average Win Rate (last 30 episodes): {avg_wins_last_30:.2%}")
    if best_params is not None:
        print("\n" + "*" * 50)  # Decorative line
        print(f"***** Best Parameters Found *****")
        print(f"* Alpha:    {best_params[0]:.4f}")
        print(f"* Gamma:    {best_params[1]:.4f}")
        print(f"* Epsilon:  {best_params[2]:.4f}")
        print(f"* Avg Reward: {best_avg_reward:.2f}")
        print("*" * 50 + "\n")  # Decorative line

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
    
    # Run Perfect Agent
    print("Running Perfect agent...")
    perfect_rewards, perfect_wins, perfect_visit_count, perfect_win_status = run_trials(PerfectAgent)

	# Train SARSA agent
    print("Training SARSA agent...")
    sarsa_rewards, sarsa_wins, sarsa_visit_count, sarsa_win_status = run_trials(SARSA_0)

    # Train Q-Learning agent
    print("Training Q-Learning agent...")
    qlearning_rewards, qlearning_wins, qlearning_visit_count, qlearning_win_status = run_trials(QLearingAgent)
    
    metrics.plot_cumulative_return(sarsa_rewards, "SARSA", qlearning_rewards, "Q-Learning")
    metrics.plot_visit_percentage("SARSA", sarsa_visit_count)
    metrics.plot_visit_percentage("Q-Learning", qlearning_visit_count)
    metrics.plot_winning_percentage("SARSA", sarsa_wins, "Q-Learning", qlearning_wins)
    metrics.plot_winning_percentage_over_episodes(sarsa_win_status, "SARSA", qlearning_win_status, "Q-Learning")
    metrics.plot_mean_visited_states_old(sarsa_visit_count, "SARSA", qlearning_visit_count, "Q-Learning")
    metrics.plot_mean_visited_states(sarsa_visit_count, "SARSA")
    metrics.plot_mean_visited_states(qlearning_visit_count, "Q-Learning")
    metrics.plot_state_action_distribution(sarsa_visit_count, "SARSA")
    metrics.plot_state_action_distribution(qlearning_visit_count, "Q-Learning")

    #verify_get_state_index(PongEnv())
    
    # Tune hyperparameters
    alpha_values = [0.01, 0.1, 0.5]  # Example learning rates
    gamma_values = [0.5, 0.9, 0.95]  # Example discount factors
    epsilon_values = [0.1, 0.2, 0.5]  # Example exploration rates

    # Run experiments for SARSA
    #run_trials_with_hyperparams(SARSA_0, alpha_values, gamma_values, epsilon_values)

    # Run experiments for Q-Learning
    run_trials_with_hyperparams(QLearingAgent, alpha_values, gamma_values, epsilon_values)

	# TODO: output and save metrics
	
