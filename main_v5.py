import os
import random
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Tic-Tac-Toe Environment Class


class TicTacToe:
    def __init__(self):
        self.board = [" " for _ in range(9)]
        self.winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]             # diagonals
        ]

    def reset(self):
        self.board = [" " for _ in range(9)]

    def is_empty(self, idx):
        return self.board[idx] == " "

    def make_move(self, idx, player):
        if self.is_empty(idx):
            self.board[idx] = player

    def check_winner(self, player):
        for combination in self.winning_combinations:
            if all([self.board[idx] == player for idx in combination]):
                return True
        return False

    def is_draw(self):
        return " " not in self.board
    def is_valid_move(self, idx, player):
        # Check if the chosen index is within the board range
        if 0 <= idx < 9:
            # Check if the chosen spot is empty
            if self.is_empty(idx):
                # Make a copy of the board to simulate the move
                temp_board = self.board[:]
                temp_board[idx] = player

                # Check if the move helps you win
                if any(all(temp_board[idx] == player for idx in combination) for combination in self.winning_combinations):
                    return "win"
                else:
                    # Check if the move blocks the opponent from winning
                    opponent = "O" if player == "X" else "X"
                    if any(all(temp_board[idx] == opponent for idx in combination) for combination in self.winning_combinations):
                        return "block"
                    else:
                        return "valid"
            else:
                return "occupied"
        else:
            return "invalid"

# Deep Q-Learning Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = 0.1  # learning rate
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration-exploitation tradeoff
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * \
            (target - self.q_table[state, action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Function for parallel training of episodes
def train_episode(episode):
    env = TicTacToe()
    state_size = 2**9
    action_size = 9
    agent_X = DQNAgent(state_size, action_size)
    agent_O = DQNAgent(state_size, action_size)

    state_X = 0
    state_O = 0
    total_reward_X = 0
    total_reward_O = 0

    # Tunable hyperparameters
    WIN_REWARD = 10
    BLOCK_REWARD = 5
    DRAW_REWARD = 2
    LOSS_PENALTY = -10
    ILLEGAL_MOVE_PENALTY = -5
    TWO_IN_A_ROW_REWARD = 2

    while True:
        # Player X's turn
        reward_O = 0
        reward_X = 0
        action_X = agent_X.choose_action(state_X)
        move_result_X = env.is_valid_move(action_X, "X")
        
        if move_result_X != "occupied" and move_result_X != "invalid":
            env.make_move(action_X, "X")
            reward_X = 0
            if env.check_winner("X"):
                reward_X += WIN_REWARD
            elif env.check_winner("O"):
                reward_X += LOSS_PENALTY
            elif env.is_draw():
                reward_X += DRAW_REWARD
            else:
                if move_result_X == "block":
                    reward_O += BLOCK_REWARD
                # Reward for two marks in a row with an empty spot on one side or in the middle
                for combination in env.winning_combinations:
                    if env.board[combination[0]] == env.board[combination[1]] == "X" and env.is_empty(combination[2]):
                        reward_X += TWO_IN_A_ROW_REWARD
        else:
            reward_X = ILLEGAL_MOVE_PENALTY

        total_reward_X += reward_X
        next_state_X = int("".join(["1" if cell == "X" else "0" for cell in env.board]), 2)
        agent_X.learn(state_X, action_X, reward_X, next_state_X, env.is_draw() or reward_X != 0)
        state_X = next_state_X

        if env.check_winner("X") or env.check_winner("O") or env.is_draw():
            break

        # Player O's turn
        action_O = agent_O.choose_action(state_O)
        move_result_O = env.is_valid_move(action_O, "O")
        
        if move_result_O != "occupied" and move_result_O != "invalid":
            env.make_move(action_O, "O")
            reward_O = 0
            if env.check_winner("O"):
                reward_O += WIN_REWARD
            elif env.check_winner("X"):
                reward_O += LOSS_PENALTY
            elif env.is_draw():
                reward_O += DRAW_REWARD
            else:
                if move_result_O == "block":
                    reward_O += BLOCK_REWARD
                
                # Reward for two marks in a row with an empty spot on one side or in the middle
                for combination in env.winning_combinations:
                    if env.board[combination[0]] == env.board[combination[1]] == "O" and env.is_empty(combination[2]):
                        reward_O += TWO_IN_A_ROW_REWARD
        else:
            reward_O = ILLEGAL_MOVE_PENALTY

        total_reward_O += reward_O
        next_state_O = int("".join(["1" if cell == "X" else "0" for cell in env.board]), 2)
        agent_O.learn(state_O, action_O, reward_O, next_state_O, env.is_draw() or reward_O != 0)
        state_O = next_state_O

        if env.check_winner("X") or env.check_winner("O") or env.is_draw():
            break

    return total_reward_X, total_reward_O


# Function to play Tic-Tac-Toe against the AI
def play_tic_tac_toe(agent_X, agent_O):
    env = TicTacToe()
    state_size = 2**9
    action_size = 9

    # Main game loop
    while True:
        # Player X's turn
        print("Current Board:")
        print(" ".join(env.board[:3]))
        print(" ".join(env.board[3:6]))
        print(" ".join(env.board[6:]))
        print("Player X's Turn")
        move_X = int(input("Enter your move (1-9): ")) - 1

        env.make_move(move_X, "X")

        # Check if Player X wins
        if env.check_winner("X"):
            print("Player X wins!")
            break

        # AI Player O's turn
        state_O = int(
            "".join(["1" if cell == "X" else "0" for cell in env.board]), 2)
        move_O = agent_O.choose_action(state_O)

        # Make the move for Player O
        env.make_move(move_O, "O")

        # Check if Player O wins
        if env.check_winner("O"):
            print("Player O wins!")
            break

        # Check for a draw
        if env.is_draw():
            print("It's a draw!")
            break

# Example usage:
# play_tic_tac_toe(agent_X, agent_O)


if __name__ == '__main__':
    # Training the DQNs using all available CPU cores
    episodes = 1000000
    print("loading")
    # Parallelize the training loop
    with ProcessPoolExecutor() as executor:
        print("ProcessPoolExecutor loaded")
        _tmp1=range(int(episodes))
        print("range(episodes) done")
        _tmp2=executor.map(train_episode, _tmp1)
        print("executor.map(train_episode, range(episodes)) done")
        tqdmclass=tqdm(_tmp2, total=int(episodes), desc="Training",)
        print("tqdm(executor.map(train_episode, range(episodes)), total=episodes, desc=\"Training\",) done")
        rewards = list(tqdmclass)
        print("list(tqdm(executor.map(train_episode, range(episodes)), total=episodes, desc=\"Training\",)) done")
    print('train finished')
    # Extract rewards for plotting
    rewards_X, rewards_O = zip(*rewards)
    print("ziped")
    # Full path to the home directory
    home_directory = os.path.expanduser("~")

    # Save the rewards as a plot in the home directory
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_X, label='Player X Rewards')
    plt.plot(rewards_O, label='Player O Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    # Save the plot as a PNG file in the home directory
    plt.savefig(os.path.join(home_directory,"xogameai", 'graph_new.png'))
    print("Training completed. Graph saved as ~/xogameai/graph_new.png.")

    # Create agent_X and agent_O instances
    state_size = 2**9
    action_size = 9
    agent_X = DQNAgent(state_size, action_size)
    agent_O = DQNAgent(state_size, action_size)

    # Save the Q-tables as numpy arrays
    np.save(os.path.join(home_directory, 'xogameai',
            'test.npy'), agent_X.q_table)
    np.save(os.path.join(home_directory, 'xogameai',
            'test.npy'), agent_O.q_table)
    print("Q-tables saved as agent_X_q_table.npy and agent_O_q_table.npy.")
    print("exit")
