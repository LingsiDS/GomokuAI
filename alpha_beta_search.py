from ast import In
from gomoku import Gomoku
from game_eval import GomokuEval, GomokuEvalOptimized
from typing import Tuple
import time
from zobrist_table import TranspositionTable
from typing import Optional


# 自定义异常，用于时间耗尽时中断搜索
class TimeUpException(Exception):
    pass


class MinmaxSearch:
    def __init__(self, time_limit: float):
        self.time_limit = time_limit
        self.tt = TranspositionTable()
        self.start_time = time.time()  # 初始化start_time属性

    def max_value(self, depth: int, alpha: float, beta: float):
        """Max value function for minimax algorithm."""
        if time.time() - self.start_time > self.time_limit:
            raise TimeUpException()

        if self.game.game_over or depth == 0:
            return GomokuEvalOptimized.evaluate(self.game)

        # 有相同局面，直接返回TT中的值
        ttv = self.probe_tt(self.game.hash_key, depth, alpha, beta)
        if ttv is not None:
            return ttv

        value, move, val = float("-inf"), None, 0
        next_moves = GomokuEval.generate_sorted_moves2(self.game)
        # TT best move first when available
        entry = self.tt.get(self.game.hash_key)
        tt_best = entry.best_move if entry else None
        if tt_best and tt_best in next_moves:
            next_moves.remove(tt_best)
            ordered = [tt_best] + next_moves
        else:
            ordered = next_moves

        for next_move in ordered:
            try:
                self.game.make_move(next_move[0], next_move[1])
                val = self.min_value(depth - 1, alpha, beta)
            finally:
                if self.game.game_over:  # 直接结束，不需要搜索其他步骤，提速明显
                    assert self.game.winner == 2, "last move is AI, AI win"
                    self.game.undo_move()
                    self.store_tt(
                        self.game.hash_key, depth, val, alpha, beta, next_move
                    )
                    return val
                self.game.undo_move()  # 确保抛出异常后也要执行undo_move

            if val > value:
                value, move = val, next_move
            alpha = max(alpha, val)
            if val >= beta:
                return value

        self.store_tt(self.game.hash_key, depth, value, alpha, beta, move)
        return value

    def min_value(self, depth: int, alpha: float, beta: float):
        """Min value function for minimax algorithm."""
        if time.time() - self.start_time > self.time_limit:
            raise TimeUpException()

        if self.game.game_over or depth == 0:
            return GomokuEvalOptimized.evaluate(self.game)

        # 有相同局面，直接返回TT中的值
        ttv = self.probe_tt(self.game.hash_key, depth, alpha, beta)
        if ttv is not None:
            return ttv

        value, move, val = float("inf"), None, 0
        next_moves = GomokuEval.generate_sorted_moves2(self.game)
        # TT best move first when available
        entry = self.tt.get(self.game.hash_key)
        tt_best = entry.best_move if entry else None
        if tt_best and tt_best in next_moves:
            next_moves.remove(tt_best)
            ordered = [tt_best] + next_moves
        else:
            ordered = next_moves

        for next_move in ordered:
            try:
                self.game.make_move(next_move[0], next_move[1])
                val = self.max_value(depth - 1, alpha, beta)
            finally:
                if self.game.game_over:  # 直接结束，不需要搜索其他步骤，提速明显
                    assert self.game.winner == 1, "last move is player, player win"
                    self.game.undo_move()
                    self.store_tt(
                        self.game.hash_key, depth, val, alpha, beta, next_move
                    )
                    return val
                self.game.undo_move()  # 确保抛出异常后也要执行undo_move

            if val < value:
                value, move = val, next_move
            beta = min(beta, val)
            if val <= alpha:
                return value

        # store in TT with remaining depth
        self.store_tt(self.game.hash_key, depth, value, alpha, beta, move)
        return value

    def minmax(self, depth: int, game: Gomoku) -> int:
        """Minimax algorithm implementation."""
        self.game = game
        val = None
        try:
            val = self.max_value(depth, float("-inf"), float("inf"))
        except TimeUpException:
            print(f"minmax search timeout at depth {depth}")
            raise TimeUpException()
        # print("minmax search val: ", val)
        return val

    def iterative_deepening(self, max_depth: int, game: Gomoku) -> Tuple:
        """简单迭代加深：从1到max_depth逐层加深"""
        self.game = game
        self.start_time = time.time()
        self.tt.new_search()

        best_move = None
        best_val = float("-inf")
        res_depth = 0

        for d in range(2, max_depth + 1, 2):
            print(f"searching depth: {d}")
            try:
                val = self.minmax(d, self.game)
                if val is not None:
                    best_val = val
                    # 直接取根节点 best_move
                    entry = self.tt.get(self.game.hash_key)
                    print(self.game.hash_key)
                    if entry and entry.best_move:
                        best_move = entry.best_move
            except TimeUpException:
                print(f"searching depth: {d}, time up, use depth {d - 2} result")
                break

        e = self.tt.get(self.game.hash_key)
        print(self.game.hash_key)
        print(f"TT entry: {e}")
        if e and e.best_move:
            best_move = e.best_move
            best_val = e.value
            res_depth = e.depth
        # if best_move is None:
        #     best_val = self.minmax(4, self.game)  # 如果没找到，则用4层搜索结果兜底
        #     best_move = None
        print(f"best move: {best_move}, depth: {res_depth}, value: {best_val}")
        return best_move

    def probe_tt(
        self, key: int, depth: int, alpha: float, beta: float
    ) -> Optional[int]:
        """
        查询 TT 是否已有局面值可以直接使用以加速搜索。

        逻辑说明：
        1. 如果 TT 中没有条目，或者存储的深度小于当前剩余搜索深度，则返回 None。
        2. 根据 flag 判断能否直接使用：
           - EXACT：值是精确值，可以直接返回。
           - LOWER：值是下界，如果 value >= beta，则可以触发 beta-cutoff，返回值。
           - UPPER：值是上界，如果 value <= alpha，则可以触发 alpha-cutoff，返回值。
        3. 其他情况返回 None，需要继续搜索。
        """
        e = self.tt.get(key)
        if e is None or e.depth < depth:
            return None
        if e.flag == TranspositionTable.EXACT:
            return e.value
        if e.flag == TranspositionTable.LOWER and e.value >= beta:
            return e.value
        if e.flag == TranspositionTable.UPPER and e.value <= alpha:
            return e.value
        return None

    def store_tt(
        self,
        key: int,
        depth: int,
        value: int,
        alpha: int,
        beta: int,
        best: Optional[Tuple[int, int]],
    ):
        """
        将当前局面的搜索结果存入 TT。

        逻辑说明：
        1. 根据 alpha-beta 的上下界判断 flag：
           - value <= alpha → UPPER (节点上界)
           - value >= beta → LOWER (节点下界)
           - 其他 → EXACT (确切值)
        2. 调用 tt.put() 保存条目，包括 depth（剩余搜索层数）、value、flag、best_move。
        3. 这样可以在下次搜索遇到相同局面时，直接用 TT 条目进行剪枝或优先扩展。
        """
        flag = TranspositionTable.EXACT
        if value <= alpha:
            flag = TranspositionTable.UPPER
        elif value >= beta:
            flag = TranspositionTable.LOWER
        self.tt.put(key, depth, value, flag, best)


class AlphaBetaSearch(MinmaxSearch):
    def __init__(self, game: Gomoku):
        super().__init__(game)

    def alpha_beta_search(self, depth: int):
        """Alpha-beta pruning algorithm implementation."""
        return self.max_value(self.game, depth, float("-inf"), float("inf"))
