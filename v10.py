import math

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def is_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all([cell == player for cell in board[i]]) or all([board[j][i] == player for j in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False

def is_draw(board):
    # Check if the game is a draw (no empty cells left)
    return all([cell != " " for row in board for cell in row])

def is_terminal(board):
    # Check if the game is in a terminal state (win, lose, or draw)
    if is_winner(board, "X"):
        return 1
    elif is_winner(board, "O"):
        return -1
    elif is_draw(board):
        return 0
    return None

def available_moves(board):
    # Return a list of available moves (empty cells) on the board
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]

def make_move(board, move, player):
    # Make a move on the board and return the new board state
    i, j = move
    new_board = [row.copy() for row in board]
    new_board[i][j] = player
    return new_board

def minimax(board, depth, maximizing_player, alpha, beta):
    terminal = is_terminal(board)
    if terminal is not None:
        return terminal * depth  # Apply depth to favor quicker wins/losses

    if maximizing_player:
        max_eval = float("-inf")
        for move in available_moves(board):
            new_board = make_move(board, move, "X")
            eval = minimax(new_board, depth + 1, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float("inf")
        for move in available_moves(board):
            new_board = make_move(board, move, "O")
            eval = minimax(new_board, depth + 1, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval

def best_move(board):
    best_score = float("-inf")
    best_move = None
    alpha = float("-inf")
    beta = float("inf")
    for move in available_moves(board):
        new_board = make_move(board, move, "X")
        eval = minimax(new_board, 0, False, alpha, beta)
        if eval > best_score:
            best_score = eval
            best_move = move
        alpha = max(alpha, eval)
    return best_move

def main():
    board = [[" " for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic-Tac-Toe!")
    print_board(board)

    while True:
        # Player's move
        while True:
            row, col = map(int, input("Enter your move (row and column): ").split())
            if board[row][col] == " ":
                board = make_move(board, (row, col), "O")
                break
            else:
                print("Invalid move. Try again.")

        print_board(board)
        result = is_terminal(board)
        if result is not None:
            if result == 1:
                print("Congratulations! You win!")
            elif result == -1:
                print("You lose! Better luck next time.")
            else:
                print("It's a draw! Good game!")
            break

        # AI's move
        print("AI is making its move...")
        row, col = best_move(board)
        board = make_move(board, (row, col), "X")

        print_board(board)
        result = is_terminal(board)
        if result is not None:
            if result == 1:
                print("You lose! Better luck next time.")
            elif result == -1:
                print("Congratulations! You win!")
            else:
                print("It's a draw! Good game!")
            break

if __name__ == "__main__":
    main()
