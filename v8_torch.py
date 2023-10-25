NUM_EPOCHS = 10000
NUM_PROCESSES = 1
BOARD_ROWS = 3
BOARD_COLS = 3
max_norm=1
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from rich import traceback,inspect
import multiprocessing,time
import multiprocessing.managers
import ctypes

class ProgBar(tqdm.tqdm):
    def update_to(self, n):
        """
        Update the progress bar in-place, useful for setting
        the state of the progress bar manually.
        E.g.:
        >>> t = tqdm(total=100) # Initialise
        >>> t.update_to(50)     # progressbar is now half-way complete
        >>> t.close()
        The last line is highly recommended, but possibly not necessary if
        `t.update_to()` will be called in such a way that `total` will be
        exactly reached and printed.

        Parameters
        ----------
        n  : int or float
            Set the internal counter of iterations.
            f using float, consider specifying `{n:.3f}`
            or similar in `bar_format`, or specifying `unit_scale`.

        """
        self.n = n
        self.refresh()

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

def train_model(model, done_train_, num_epochs=10000):
    traceback.install()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Experiment with different learning rates
    losses = []
    for epoch in range(num_epochs):
        p1 = Player("p1")
        p2 = Player("p2")
        st = State(p1, p2)

        while not st.isEnd:
            positions = st.availablePositions()
            curr_board = torch.tensor(st.board).float().view(1, -1)
            curr_board.requires_grad = False
            action = p1.chooseAction(positions, curr_board, st.playerSymbol)
            st.updateState(action)
            board_hash = st.getHash()
            p1.addState(board_hash)
            win = st.winner()
            if win is not None:
                st.giveReward()
                p1.reset()
                p2.reset()
                st.reset()
                break
            else:
                positions = st.availablePositions()
                curr_board = torch.tensor(st.board).float().view(1, -1)
                action = p2.chooseAction(positions, curr_board, st.playerSymbol)
                st.updateState(action)
                board_hash = st.getHash()
                p2.addState(board_hash)
                win = st.winner()
                if win is not None:
                    st.giveReward()
                    p1.reset()
                    p2.reset()
                    st.reset()
                    break

        # Training the model
        states = torch.tensor(p1.states).float().view(-1, BOARD_ROWS * BOARD_COLS)
        rewards = torch.tensor(p1.rewards).float().view(-1, 1)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        prediction = model(states)
        loss = nn.BCEWithLogitsLoss()(prediction, rewards)  # Using BCEWithLogitsLoss directly
        optimizer.zero_grad()

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        for param in model.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, max_norm)  # Apply gradient clipping
        # loss.backward()
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("NaN gradient detected!")

        # Clip gradients per parameter
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = torch.clamp(param.grad.data, -1, 1)  # Clip gradients to the range [-1, 1]

        # optimizer.step()

        optimizer.step()

        # Monitoring progress
        if epoch % 10000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
        losses.append(loss.item())
        done_train_.value += 1
    # print('Training finished!')
    # torch.save(model.state_dict(), 'model.pth')

    # Plotting the loss graph and saving it as graph_loss.png
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, color='blue', marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('graph_loss.png')
    # plt.show()

def train_parallel(model, num_epochs, process_number,shared_var):
    mod=model[0]
    # print(f"Process {process_number} started training...")
    traceback.install()
    train_model(mod,shared_var, num_epochs)
    # print(f"Process {process_number} finished training!")

def update_prog(shared_var):
    prog=ProgBar(total=NUM_EPOCHS,desc="Training", position=0)
    while shared_var.value != NUM_EPOCHS:
        prog.update_to(shared_var.value)
        time.sleep(0.3)
if __name__ == "__main__":
    traceback.install()
    # Number of parallel processes
    processes = []
    shared_var=multiprocessing.Value('i',0)
    manager = multiprocessing.Manager()
    model = manager.list([TicTacToeNet()])
    # inspect(model,all=True)
    for i in range(NUM_PROCESSES):
        process = multiprocessing.Process(target=train_parallel, args=(model, NUM_EPOCHS // NUM_PROCESSES, i,shared_var))
        processes.append(process)
        process.start()
    # train_parallel(model, NUM_EPOCHS // NUM_PROCESSES, 0,shared_var)
    multiprocessing.Process(target=update_prog,args=(shared_var,)).start()
    for process in processes:
        process.join()
    print("All processes finished training!")
    print("saving")
    torch.save(model[0].state_dict(), 'model.pth')