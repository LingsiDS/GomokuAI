from gomoku import Gomoku
from game_eval import GomokuEval
from typing import Tuple


class MinmaxSearch:
    def __init__(self):
        pass

    def max_value(self, depth: int):
        """Max value function for minimax algorithm."""
        if self.game.game_over or depth == 0:
            return GomokuEval.evaluate(self.game), None, float("-inf"), float("inf")

        alpha, beta = float("-inf"), float("inf")
        value, move = float("-inf"), None
        next_moves = GomokuEval.generate_moves(self.game)
        for next_move in next_moves:
            self.game.make_move(next_move[0], next_move[1])
            val, mv, a, b = self.min_value(depth - 1)
            self.game.undo_move(next_move[0], next_move[1])
            if val > value:
                value, move = val, next_move
            alpha = max(alpha, val)
            if val >= b:
                return value, move, alpha, beta
        return value, move, alpha, beta

    def min_value(self, depth: int):
        """Min value function for minimax algorithm."""
        if self.game.game_over or depth == 0:
            return GomokuEval.evaluate(self.game), None, float("-inf"), float("inf")

        alpha, beta = float("-inf"), float("inf")
        next_moves = GomokuEval.generate_moves(self.game)
        value, move = float("inf"), None
        for next_move in next_moves:
            self.game.make_move(next_move[0], next_move[1])
            val, mv, a, b = self.max_value(depth - 1)
            self.game.undo_move(next_move[0], next_move[1])
            if val < value:
                value, move = val, next_move
            beta = min(beta, val)
            if val <= a:
                return value, move, alpha, beta
        return value, move, alpha, beta

    def minmax(self, depth: int, game: Gomoku) -> Tuple:
        """Minimax algorithm implementation."""
        self.game = game
        _, move, _, _ = self.max_value(depth)
        return move


class AlphaBetaSearch(MinmaxSearch):
    def __init__(self, game: Gomoku):
        super().__init__(game)

    def alpha_beta_search(self, depth: int):
        """Alpha-beta pruning algorithm implementation."""
        return self.max_value(self.game, depth, float("-inf"), float("inf"))
