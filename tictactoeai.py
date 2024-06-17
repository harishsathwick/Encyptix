

import copy

# Définition des symboles pour les joueurs
PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '
     

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("__" * 5)
     

def is_winner(board, player):
    # Vérifier les lignes, colonnes et diagonales
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or \
           all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or \
       all(board[i][2 - i] == player for i in range(3)):
        return True
    return False
     

def is_board_full(board):
    return all(board[i][j] != EMPTY for i in range(3) for j in range(3))
     

def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]
     

def minimax(board, depth, maximizing_player):
    if is_winner(board, PLAYER_X):
        return -1
    elif is_winner(board, PLAYER_O):
        return 1
    elif is_board_full(board):
        return 0

    if maximizing_player:
        max_eval = float('-inf')
        for i, j in get_empty_cells(board):
            board[i][j] = PLAYER_O
            eval = minimax(board, depth + 1, False)
            board[i][j] = EMPTY
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i, j in get_empty_cells(board):
            board[i][j] = PLAYER_X
            eval = minimax(board, depth + 1, True)
            board[i][j] = EMPTY
            min_eval = min(min_eval, eval)
        return min_eval
     

def find_best_move(board):
    best_val = float('-inf')
    best_move = (-1, -1)

    for i, j in get_empty_cells(board):
        board[i][j] = PLAYER_O
        move_val = minimax(board, 0, False)
        board[i][j] = EMPTY

        if move_val > best_val:
            best_move = (i, j)
            best_val = move_val

    return best_move

     

def play_tic_tac_toe():
    # Initialiser le plateau
    board = [[EMPTY] * 3 for _ in range(3)]

    while True:
        print_board(board)

        # Tour du joueur humain
        row = int(input("Enter row (0, 1, or 2): "))
        col = int(input("Enter column (0, 1, or 2): "))

        if board[row][col] == EMPTY:
            board[row][col] = PLAYER_X
        else:
            print("Cell already occupied. Try again.")
            continue

        if is_winner(board, PLAYER_X):
            print("You win!")
            break
        elif is_board_full(board):
            print("It's a tie!")
            break

        # Tour de l'IA
        print("AI's turn:")
        print("__" * 5)
        best_move = find_best_move(board)
        board[best_move[0]][best_move[1]] = PLAYER_O

        if is_winner(board, PLAYER_O):
            print_board(board)
            print("AI wins!")
            break
        elif is_board_full(board):
            print_board(board)
            print("It's a tie!")
            break

if __name__ == "__main__":
    play_tic_tac_toe()

     
 