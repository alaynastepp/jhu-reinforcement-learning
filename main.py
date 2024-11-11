import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from basicPongEnv import PongEnv
#from exampleAgent import Agent
from alaynaEnv import PongEnv
from QLearning_agent import QLearingAgent
from testAgent import TestAgent
from pongVisualizer import PongVisualizer
import metrics

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

    metrics.pretty_print_metrics(avg_rewards_last_30, avg_wins_last_30, percentage_visited_0_to_63, percentage_visited_64_to_127)

    return avg_rewards, avg_wins, visit_count, np.mean(all_wins, axis=0)

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

	# Run SARSA agent
    #print("Training SARSA agent...")
    #avg_rewards = run_experiment(SARSA_0)
    
    print("Training Test agent...")
    test_rewards, test_wins, test_visit_count, test_win_status = run_trials(TestAgent)

    # Run Q-Learning agent
    print("Training Q-Learning agent...")
    qlearning_rewards, qlearning_wins, qlearning_visit_count, qlearning_win_status = run_trials(QLearingAgent)
    
    metrics.plot_cumulative_return(test_rewards, qlearning_rewards)
    metrics.plot_visit_percentage("test", test_visit_count)
    metrics.plot_visit_percentage("Q-Learning", qlearning_visit_count)
    metrics.plot_winning_percentage(test_wins, qlearning_wins)
    metrics.plot_winning_percentage_over_episodes(test_win_status, qlearning_win_status)
    metrics.plot_mean_visited_states_old(test_visit_count, qlearning_visit_count)
    metrics.plot_mean_visited_states(test_visit_count, "test")
    metrics.plot_mean_visited_states(qlearning_visit_count, "Q-Learning")
    metrics.plot_state_action_distribution(test_visit_count, "test")
    metrics.plot_state_action_distribution(qlearning_visit_count, "Q-Learning")
    
    #verify_get_state_index(PongEnv())

	# TODO: output and save metrics
	
