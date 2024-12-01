"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    X_count = sum(row.count(X) for row in board)
    O_count = sum(row.count(O) for row in board)
    if X_count == O_count:
        return X
    return O
    


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return set([
        (i, j)
        for i in range(3)
        for j in range(3)
        if board[i][j] == EMPTY
    ])


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = copy.deepcopy(board)
    if new_board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid action.")
    else:
        new_board[action[0]][action[1]] = player(new_board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = [
        [board[i][0] for i in range(3)],
        [board[i][1] for i in range(3)],
        [board[i][2] for i in range(3)],
        board[0],
        board[1],
        board[2],
        [board[i][i] for i in range(3)],
        [board[2 - i][i] for i in range(3)],
    ]

    for line in lines:
        if line.count(X) == 3:
            return X
        if line.count(O) == 3:
            return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    return all(EMPTY not in row for row in board)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    #Define a função auxiliar para calcular o valor máximo de uma jogada
    def maxValue(board):
        #Caso o jogo termine, retorna a utilidade do estado final (vitória, derrota ou empate)
        if terminal(board):
            return utility(board)
        #Retorna o maior valor entre as opções de ação possíveis para o jogador 'X'
        return max(minValue(result(board, action)) for action in actions(board))

    #Define a função auxiliar para calcular o valor mínimo de uma jogada
    def minValue(board):
        #Caso o jogo termine, retorna a utilidade do estado final
        if terminal(board):
            return utility(board)
        #Retorna o menor valor entre as opções de ação possíveis para o jogador 'O'
        return min(maxValue(result(board, action)) for action in actions(board))

    #Se o jogo já terminou no estado atual, não há ações possíveis
    if terminal(board):
        return None

    #Determina o jogador atual ('X' ou 'O')
    tourn = player(board)

    #Inicializa variáveis para armazenar a melhor jogada e seu valor
    if tourn == X:
        #Caso o jogador atual seja 'X', busca maximizar o valor
        value = -math.inf  #Valor inicial baixo para maximizar
        move = None        #Variável para armazenar a melhor jogada
        #Itera sobre todas as ações possíveis
        for action in actions(board):
            #Calcula o valor mínimo resultante da jogada
            minValueResult = minValue(result(board, action))
            #Se o valor atual for maior que o melhor valor até agora, atualiza o melhor valor e a jogada
            if minValueResult > value:
                value = minValueResult
                move = action
    else:
        #Caso o jogador atual seja 'O', busca minimizar o valor
        value = math.inf  #Valor inicial alto para minimizar
        move = None       #Variável para armazenar a melhor jogada
        #Itera sobre todas as ações possíveis
        for action in actions(board):
            #Calcula o valor máximo resultante da jogada
            maxValueResult = maxValue(result(board, action))
            #Se o valor atual for menor que o melhor valor até agora, atualiza o melhor valor e a jogada
            if maxValueResult < value:
                value = maxValueResult
                move = action

    #Retorna a melhor jogada encontrada
    return move
