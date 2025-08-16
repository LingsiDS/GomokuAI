from gomoku import Gomoku
from game_eval import GomokuEval, GomokuEvalOptimized
from typing import Tuple


class MinmaxSearch:
    def __init__(self):
        pass

    def max_value(self, depth: int, alpha: float, beta: float):
        """Max value function for minimax algorithm."""
        if self.game.game_over or depth == 0:
            return GomokuEvalOptimized.evaluate(self.game), None
        if self.game.check_last_move_win():  # last move is player, player win
            assert self.game.current_player == 1, "last move is player, player win"
            return float("-inf"), None

        value, move = float("-inf"), None
        next_moves = GomokuEval.generate_sorted_moves(self.game)
        for next_move in next_moves:
            game_over = self.game.make_move(next_move[0], next_move[1])
            val, mv = self.min_value(depth - 1, alpha, beta)
            self.game.undo_move()
            if game_over:  # 直接结束，不需要搜索其他步骤，提速明显
                return val, next_move
            if val > value:
                value, move = val, next_move
            alpha = max(alpha, val)
            if val >= beta:
                return value, move
        return value, move

    def min_value(self, depth: int, alpha: float, beta: float):
        """Min value function for minimax algorithm."""
        if self.game.game_over or depth == 0:
            return GomokuEvalOptimized.evaluate(self.game), None
        if self.game.check_last_move_win():  # last move is AI, AI win
            assert self.game.current_player == 2, "last move is AI, AI win"
            return float("inf"), None

        next_moves = GomokuEval.generate_sorted_moves(self.game)
        value, move = float("inf"), None
        for next_move in next_moves:
            game_over = self.game.make_move(next_move[0], next_move[1])
            val, mv = self.max_value(depth - 1, alpha, beta)
            self.game.undo_move()
            if game_over:  # 直接结束，不需要搜索其他步骤，提速明显
                return val, next_move
            if val < value:
                value, move = val, next_move
            beta = min(beta, val)
            if val <= alpha:
                return value, move
        return value, move

    def minmax(self, depth: int, game: Gomoku) -> Tuple:
        """Minimax algorithm implementation."""
        self.game = game
        val, move = self.max_value(depth, float("-inf"), float("inf"))
        # print("minmax search val: ", val)
        return move


class AlphaBetaSearch(MinmaxSearch):
    def __init__(self, game: Gomoku):
        super().__init__(game)

    def alpha_beta_search(self, depth: int):
        """Alpha-beta pruning algorithm implementation."""
        return self.max_value(self.game, depth, float("-inf"), float("inf"))
