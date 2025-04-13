class NineMenMorris:
    adjacency = {
        0: [1, 9], 1: [0, 2, 4], 2: [1, 14], 3: [4, 10], 4: [1, 3, 5, 7],
        5: [4, 13], 6: [7, 11], 7: [4, 6, 8], 8: [7, 12], 9: [0, 10, 21],
        10: [3, 9, 11, 18], 11: [6, 10, 15], 12: [8, 13, 17],
        13: [5, 12, 14, 20], 14: [2, 13, 23], 15: [11, 16],
        16: [15, 17, 19], 17: [12, 16], 18: [10, 19],
        19: [16, 18, 20, 22], 20: [13, 19], 21: [9, 22],
        22: [19, 21, 23], 23: [14, 22]
    }
    mills = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
        [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23],
        [0, 9, 21], [3, 10, 18], [6, 11, 15], [1, 4, 7],
        [16, 19, 22], [8, 12, 17], [5, 13, 20], [2, 14, 23]
    ]

    def __init__(self, player_one_pieces: int, player_two_pieces: int):
        self.board = [0] * 24
        self.player_pieces = {1: player_one_pieces, 2: player_two_pieces}

    def is_mill(self, position, player):
        for mill in self.mills:
            if position in mill and all(self.board[pos] == player for pos in mill):
                return True
        return False

    def valid_moves(self, position):
        return self.adjacency.get(position, [])

    def place_piece(self, position, player):
        if self.board[position] == 0 and self.player_pieces[player] > 0:
            self.board[position] = player
            self.player_pieces[player] -= 1
            return True
        return False

    def move_piece(self, start, end, player):
        if self.board[start] == player and self.board[end] == 0 and end in self.valid_moves(start):
            self.board[start] = 0
            self.board[end] = player
            return True
        return False


class MinMaxAlgorithm:
    def __init__(self, game):
        self.game = game

    def evaluate_board(self):
        score = 0
        for mill in self.game.mills:
            player_one_count = sum(1 for pos in mill if self.game.board[pos] == 1)
            player_two_count = sum(1 for pos in mill if self.game.board[pos] == 2)
            if player_one_count == 3:
                score += 10
            elif player_two_count == 3:
                score -= 10
        return score

    def minmax(self, depth, maximizing_player):
        if depth == 0:
            return self.evaluate_board()
        if maximizing_player:
            max_evaluation = float('-inf')
            for position in range(24):
                if self.game.board[position] == 0:
                    self.game.board[position] = 1
                    evaluation = self.minmax(depth - 1, False)
                    self.game.board[position] = 0
                    max_evaluation = max(max_evaluation, evaluation)
            return max_evaluation
        else:
            min_evaluation = float('inf')
            for position in range(24):
                if self.game.board[position] == 0:
                    self.game.board[position] = 2
                    evaluation = self.minmax(depth - 1, True)
                    self.game.board[position] = 0
                    min_evaluation = min(min_evaluation, evaluation)
            return min_evaluation


class AlphaBetaAlgorithm:
    def __init__(self, game):
        self.game = game

    def evaluate_board(self):
        score = 0
        for mill in self.game.mills:
            player_one_count = sum(1 for pos in mill if self.game.board[pos] == 1)
            player_two_count = sum(1 for pos in mill if self.game.board[pos] == 2)
            if player_one_count == 3:
                score += 10
            elif player_two_count == 3:
                score -= 10
        return score

    def alphabeta(self, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.evaluate_board()
        if maximizing_player:
            maximum_evaluation = float('-inf')
            for position in range(24):
                if self.game.board[position] == 0:
                    self.game.board[position] = 1
                    evaluation = self.alphabeta(depth - 1, alpha, beta, False)
                    self.game.board[position] = 0
                    maximum_evaluation = max(maximum_evaluation, evaluation)
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        break
            return maximum_evaluation
        else:
            minimum_evaluation = float('inf')
            for position in range(24):
                if self.game.board[position] == 0:
                    self.game.board[position] = 2
                    evaluation = self.alphabeta(depth - 1, alpha, beta, True)
                    self.game.board[position] = 0
                    minimum_evaluation = min(minimum_evaluation, evaluation)
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        break
            return minimum_evaluation


def simulate_optimal_moves(initial_state, player_pieces, algorithm_name, depth):
    game = NineMenMorris(player_pieces[0], player_pieces[1])
    game.board = initial_state.copy()

    if algorithm_name == "MinMax":
        algorithm = MinMaxAlgorithm(game)
    elif algorithm_name == "AlphaBeta":
        algorithm = AlphaBetaAlgorithm(game)
    else:
        raise ValueError("Invalid algorithm name")

    states = [initial_state.copy()]
    current_player = 1

    while True:
        best_move = None
        best_score = float('-inf') if current_player == 1 else float('inf')

        for position in range(24):
            if game.board[position] == 0:
                game.board[position] = current_player
                if algorithm_name == "MinMax":
                    score = algorithm.minmax(depth - 1, current_player == 2)
                else:
                    score = algorithm.alphabeta(depth - 1, float('-inf'), float('inf'), current_player == 2)
                game.board[position] = 0

                if (current_player == 1 and score > best_score) or (current_player == 2 and score < best_score):
                    best_score = score
                    best_move = position

        if best_move is not None and game.place_piece(best_move, current_player):
            states.append(game.board.copy())
        else:
            break

        current_player = 3 - current_player

    return states


#initial_values = ['', '', '', '', '', 'x', '', '', '0', '', 'x', 'x', '0', 'x', '', '', '', '0', 'x', '0', 'x', '', '0', '']
# output = simulate_optimal_moves(initial_values, (3, 4), 'MinMax', 3)

# Nu trebuie oferite justificari suplimentare pentru aceasta problema,
# am aplicat algoritmii MinMax si AlphaBeta asa cum sunt ei descrisi
# in sectiune de teorie din lab 4