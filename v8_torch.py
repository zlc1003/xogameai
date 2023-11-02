import torch
import numpy as np
from rich import traceback
traceback.install()

BOARD_ROWS = 3
BOARD_COLS = 3

class TicTacToeNet(torch.nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc = torch.nn.Linear(BOARD_ROWS * BOARD_COLS, 1)

    def forward(self, x):
        x = x.view(-1, BOARD_ROWS * BOARD_COLS)
        x = torch.sigmoid(self.fc(x))  # Keep the sigmoid activation
        return x

class TicTacToeGame:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.player_symbol = 1  # You are "X"
        self.computer_symbol = -1  # The trained model is "O"
        self.empty_symbol = 0
        self.model = TicTacToeNet()
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()

    def print_board(self):
        symbols = [' ', 'X', 'O']
        for r in range(BOARD_ROWS):
            row = [symbols[int(self.board[r][c])] for c in range(BOARD_COLS)]
            print("|".join(row))
            if r < BOARD_ROWS - 1:
                print("-" * (BOARD_COLS * 2 - 1))

    def is_valid_move(self, move):
        r, c = move
        return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS and self.board[r][c] == self.empty_symbol

    def make_move(self, move, symbol):
        r, c = move
        self.board[r][c] = symbol

    def player_turn(self):
        while True:
            try:
                move = input("Enter your move (row and column, e.g., '1 2'): ").split()
                move = (int(move[0]), int(move[1]))
                if self.is_valid_move(move):
                    return move
                else:
                    print("Invalid move. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter row and column values.")

    def computer_turn(self):
        available_positions = [(r, c) for r in range(BOARD_ROWS) for c in range(BOARD_COLS) if self.board[r][c] == self.empty_symbol]
        if available_positions:
            current_board = torch.tensor(self.board).float().view(1, -1)
            action = self.computer_choose_action(available_positions, current_board)
            return action

    def computer_choose_action(self, positions, current_board):
        available_positions = [(r, c) for r, c in positions]
        current_board = current_board.view(-1, BOARD_ROWS * BOARD_COLS)
        value_max = -999
        action = None
        for p in positions:
            next_board = current_board.clone()
            next_board[0, p[0] * BOARD_COLS + p[1]] = self.computer_symbol
            next_boardHash = str(next_board.view(BOARD_ROWS * BOARD_COLS).int().tolist())
            value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
            if value >= value_max:
                value_max = value
                action = p
        return action

    def play_game(self):
        self.print_board()
        while True:
            player_move = self.player_turn()
            self.make_move(player_move, self.player_symbol)
            self.print_board()
            if self.check_winner(self.player_symbol):
                print("You win! Congratulations!")
                break
            if self.check_draw():
                print("It's a draw!")
                break

            computer_move = self.computer_turn()
            if computer_move:
                self.make_move(computer_move, self.computer_symbol)
                self.print_board()
                if self.check_winner(self.computer_symbol):
                    print("Computer wins!")
                    break
                if self.check_draw():
                    print("It's a draw!")
                    break

    def check_winner(self, symbol):
        for i in range(BOARD_ROWS):
            if all(self.board[i, :] == symbol) or all(self.board[:, i] == symbol):
                return True
        if all(np.diag(self.board) == symbol) or all(np.diag(np.fliplr(self.board)) == symbol):
            return True
        return False

    def check_draw(self):
        return np.all(self.board != self.empty_symbol)

if __name__ == "__main__":
    game = TicTacToeGame()
    print("Welcome to Tic-Tac-Toe!")
    game.play_game()