import torch
import torch.nn as nn
BOARD_ROWS = 3
BOARD_COLS = 3
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc = nn.Linear(BOARD_ROWS * BOARD_COLS, 1)

    def forward(self, x):
        x = x.view(-1, BOARD_ROWS * BOARD_COLS)
        x = torch.sigmoid(self.fc(x))  # Keep the sigmoid activation
        return x

# Function to check if a player has won
def check_winner(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or \
           all(board[j][i] == player for j in range(3)) or \
           all(board[j][j] == player for j in range(3)) or \
           all(board[j][2-j] == player for j in range(3)):
            return True
    return False

# Function to print the Tic-Tac-Toe board
def print_board(board):
    for row in board:
        print(" | ".join(["X" if cell == 1 else "O" if cell == -1 else " " for cell in row]))
        print("-" * 9)

# Function to get the best move from the AI
def get_best_move(board, model):
    board_tensor = board.view(1, 1, 3, 3).float()  # Add batch and channel dimensions
    with torch.no_grad():
        prediction = model(board_tensor)
    mapped_move = int(torch.clamp(prediction, 0, 8).item())
    row = mapped_move // 3
    col = mapped_move % 3
    print("AI's raw predictions:", prediction)
    print("AI's mapped move (row, col):", row, col)
    return row, col

# Main game loop
def play_game(model):
    board = torch.zeros(3, 3)  # Initialize an empty 3x3 board
    print("Welcome to Tic-Tac-Toe!")
    print_board(board)
    
    while True:
        # Player's turn
        row, col = map(int, input("Enter your move (row [0-2] and column [0-2] separated by space): ").split())
        if board[row][col] != 0:
            print("Invalid move. Try again.")
            continue
        board[row][col] = -1
        print_board(board)
        
        # Check if the player wins
        if check_winner(board, -1):
            print("Congratulations! You win!")
            break
        
        # AI's turn
        print("AI is thinking...")
        ai_row, ai_col = get_best_move(board, model)
        board[ai_row][ai_col] = 1
        print_board(board)
        
        # Check if the AI wins
        if check_winner(board, 1):
            print("AI wins! Better luck next time.")
            break
        
        # Check for a draw
        if not (board == 0).any():
            print("It's a draw!")
            break

# Start the game
model = TicTacToeNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Start the game
play_game(model)
