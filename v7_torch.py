import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm
import matplotlib.pyplot as plt

# Define the TicTacToeNet model with Batch Normalization
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 3 * 3, 1)  # Output layer for regression

    def forward(self, x):
        x = x.view(-1, 1, 3, 3)  # Add channel dimension for convolution
        # Apply Batch Normalization and ReLU activation
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = torch.relu(self.batch_norm4(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)
        return x

# Function to get rewards based on game outcome
def get_reward(winner):
    clipped_reward = max(min(winner, 1), -1)
    if clipped_reward == 1:
        return 1.0, -1.0  # Player 1 wins, Player 2 loses
    elif clipped_reward == -1:
        return -1.0, 1.0  # Player 1 loses, Player 2 wins
    else:
        return 0.1, 0.1  # Draw, both players get a small reward

# Function to get the best move using the model and exploration strategy
def get_best_move(board, model, temperature=3):
    board_tensor = board.view(1, 1, 3, 3).float()  # Add batch and channel dimensions
    with torch.no_grad():
        raw_predictions = model(board_tensor)
    
    # Apply temperature-based exploration
    scaled_predictions = raw_predictions / temperature
    
    # Convert raw predictions to probabilities using softmax function
    probabilities = torch.softmax(scaled_predictions, dim=1)
    
    # Sample a move from the probability distribution
    move_idx = torch.multinomial(probabilities.view(-1), 1).item()
    row = move_idx // 3
    col = move_idx % 3
    print("AI's raw predictions:", raw_predictions)
    print("AI's scaled predictions:", scaled_predictions)
    print("AI's mapped move (row, col):", row, col)
    return row, col



isEnd=0

# Function to check the game winner and game completion status
def winner(board):
    global isEnd
    BOARD_ROWS = [board[0], board[1], board[2]]
    BOARD_COLS = [[board[0][0], board[1][0], board[2][0]],
                  [board[0][1], board[1][1], board[2][1]],
                  [board[0][2], board[1][2], board[2][2]]]
    # Check rows
    for i in range(len(BOARD_ROWS)):
        if sum(BOARD_ROWS[i]) == 3:
            isEnd = True
            return 1
        if sum(BOARD_ROWS[i]) == -3:
            isEnd = True
            return -1
    # Check columns
    for i in range(len(BOARD_COLS)):
        if sum(board[:, i]) == 3:
            isEnd = True
            return 1
        if sum(board[:, i]) == -3:
            isEnd = True
            return -1
    # Check diagonals
    diag_sum1 = sum([board[i, i] for i in range(3)])
    diag_sum2 = sum([board[i, 2 - i] for i in range(3)])
    diag_sum = max(abs(diag_sum1), abs(diag_sum2))
    if diag_sum == 3:
        isEnd = True
        if diag_sum1 == 3 or diag_sum2 == 3:
            return 1
        else:
            return -1

    # Check for a draw (no available positions)
    if len(empty_positions(board)) == 0:
        isEnd = True
        return 0
    # Game not ended yet
    isEnd = False
    return None

# Function to get empty positions on the board
def empty_positions(board):
    positions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                positions.append((i, j))
    return positions

# Function to check if the board is valid (no NaN or Inf values)
def check_valid_board(board):
    if torch.isnan(board).any() or torch.isinf(board).any():
        return False
    return True

# Function to check if a move is valid on the board
def is_valid_move(board, row, col):
    """
    Check if the given move (row, col) is valid on the board.
    """
    if -1<row<3 and -1<col<3:
        if board[row][col] == 0:
            return True
        else:
            # AI made an invalid move, give a punishment and end the round
            return False
    else:
        # AI made an invalid move, give a punishment and end the round
        return False

# Function to play the game using the model
def play_game(model):
    board = torch.zeros(3, 3)  # Initialize an empty 3x3 board
    while True:
        row, col = get_best_move(board, model)
        if not is_valid_move(board, row, col):
            return -10, 10  # AI made an invalid move, return rewards (-10, 10)
        board[row][col] = 1
        if winner(board) == 1:
            return 10, -10  # Player 1 wins, return rewards (10, -10)
        if not (board == 0).any():
            return 0, 0  # Draw, return rewards (0, 0)

# Define the number of training epochs and maximum gradient norm
num_epochs = 10000
max_grad_norm = 1.0  # Set your desired maximum gradient norm value
model = TicTacToeNet()

# Normalization parameters
mean = 0.5
std = 0.5
losses = []
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust learning rate
board = torch.zeros(3, 3)
normalized_board = (board - mean) / std  # Normalize the initial empty board

# Training loop
for epoch in tqdm.tqdm(range(num_epochs), desc="Training", position=0):
    player1_reward, player2_reward = play_game(model)

    if not check_valid_board(normalized_board):
        print("Invalid game board encountered during self-play!")
        continue

    # Normalize the board before passing it to the model
    normalized_board = (board - mean) / std

    optimizer.zero_grad()
    outputs = model(normalized_board.view(1, 1, 3, 3).float())
    loss = outputs.mean()  # You might want to define an appropriate loss function

    # Use get_reward to get rewards for players 1 and 2
    player1_loss, player2_loss = get_reward(player1_reward)

    # Apply rewards to the loss function
    loss += player1_loss * player1_reward
    loss += player2_loss * player2_reward

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm)  # Clip gradients
    optimizer.step()

    # Monitoring progress and saving losses
    losses.append(loss.item())
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Save the loss graph
plt.figure(figsize=(10, 6))
plt.plot(losses, color='b', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('graph_loss.png')
plt.show()

print('Training finished!')
torch.save(model.state_dict(), 'model.pth')
