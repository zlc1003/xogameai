import torch
from rich import traceback
import numpy as np
BOARD_ROWS = 3
BOARD_COLS = 3
traceback.install()
max_norm=1
import torch
import torch.nn as nn
import tqdm
class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            try:
                action_index = int(input("Enter the index of the position you want to go: "))
                if action_index in range(len(positions)):
                    return positions[action_index]
                else:
                    print("Invalid index. Please choose a valid position.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc = nn.Linear(BOARD_ROWS * BOARD_COLS, 1)

    def forward(self, x):
        x = x.view(-1, BOARD_ROWS * BOARD_COLS)
        x = torch.sigmoid(self.fc(x))  # Keep the sigmoid activation
        return x

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.playerSymbol = 1

    def getHash(self):
        return str(self.board.reshape(BOARD_COLS * BOARD_ROWS))

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        # col
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in tqdm.tqdm(range(rounds)):
            # if i % 1000 == 0:
            #     print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    # Player 2
                    positions = self.availablePositions()
                    curr_board = torch.tensor(self.board).float()
                    p2_action = self.p2.chooseAction(positions, curr_board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.rewards = []
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash


    # append a hash state
    def addState(self, state):
        self.states.append(state)  # add states to the list

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        # Normalize rewards
        rewards = torch.tensor(self.rewards).float()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Create a copy of rewards before iterating through it
        rewards_copy = rewards.clone()

        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            reward_value = rewards_copy.mean().item()  # Calculate the mean of rewards and obtain a scalar value
            self.states_value[st] += self.lr * (self.decay_gamma * reward_value - self.states_value[st])

            # Remove the last reward value from the list
            rewards_copy = rewards_copy[:-1]

        # Clear the rewards list after updating states values
        self.rewards = []



    def reset(self):
        self.states = []

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.clone()
                next_board[0, p] = symbol  # modify the cloned tensor
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action
if __name__ == "__main__":
    # Load the saved model and create a human player
    model = TicTacToeNet()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set the model to evaluation mode
    human_player = HumanPlayer("Human")

    # Game loop
    while True:
        st = State(Player("AI1"), Player("AI2"))  # Provide names for players

        # Human player's turn
        while not st.isEnd and st.playerSymbol == 1:
            positions = st.availablePositions()
            human_action = human_player.chooseAction(positions)
            st.updateState(human_action)
            board_hash = st.getHash()
            human_player.addState(board_hash)

        # AI player's turn
        while not st.isEnd and st.playerSymbol == -1:
            positions = st.availablePositions()
            curr_board = torch.tensor(st.board).float().view(1, -1)
            action_prob = model(curr_board)
            print("Action Probabilities:", action_prob)
            print("Available Positions:", positions)
            valid_moves = [positions[i] for i in range(len(positions)) if st.board[positions[i]] == 0]
            print("Valid Moves:", valid_moves)
            action_probs_valid = action_prob[0, [positions.index(move) for move in valid_moves]]
            action_index = torch.argmax(action_probs_valid).item()
            ai_action = valid_moves[action_index]
            st.updateState(ai_action)
            board_hash = st.getHash()
            # Assuming it's Player 2's turn, update the state for Player 2 (modify accordingly if needed)
            st.p2.addState(board_hash)


        # Check game result and update rewards (modify this part based on your reward mechanism)
        result = st.winner()
        if result == 1:
            human_player.feedReward(1)
        elif result == -1:
            human_player.feedReward(-1)
        else:
            human_player.feedReward(0)

        # Reset players and board for the next game
        human_player.reset()
        st.reset()

        # Ask if the user wants to play again
        play_again = input("Play again? (yes/no): ")
        if play_again.lower() != "yes":
            break
