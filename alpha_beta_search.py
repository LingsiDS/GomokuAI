from ast import In
from gomoku import Gomoku
from game_eval import GomokuEval, GomokuEvalOptimized
from typing import Tuple
import time
from zobrist_table import TranspositionTable
from typing import Optional, List


# 自定义异常，用于时间耗尽时中断搜索
class TimeUpException(Exception):
    pass


class AlphaBetaSearch:
    def __init__(self, time_limit: float):
        self.time_limit = time_limit
        self.tt = TranspositionTable()
        self.start_time = time.time()  # 初始化start_time属性

    def max_value(
        self, depth: int, alpha: float, beta: float, debug_path: Optional[List] = None
    ) -> Tuple[int, Optional[List[Tuple[int, int]]]]:
        """Max value函数，返回 (评估分, 导致该分数的路径)。"""
        if time.time() - self.start_time > self.time_limit:
            raise TimeUpException()

        # 到达搜索尽头（叶子节点）
        if self.game.game_over or depth == 0:
            score = GomokuEvalOptimized.evaluate(self.game)
            # 如果在调试模式下，打印最终路径、得分和棋盘
            if debug_path is not None:
                print(f"\n  [LEAF NODE] Path: {debug_path}")
                print(f"  >>> Leads to Score: {score}")
                self.game.print_board()  # 打印最终棋盘状态
            return score, []

        ttv, tt_path = self.probe_tt(
            self.game.hash_key, depth, alpha, beta
        )  # 假设probe_tt也返回路径
        if ttv is not None:
            return ttv, tt_path

        value, move = float("-inf"), None
        best_path_for_this_node = []

        ordered = GomokuEval.generate_sorted_moves2(self.game)

        for next_move in ordered:
            response_path = []
            try:
                if debug_path is not None:
                    debug_path.append(next_move)

                assert self.game.make_move(next_move[0], next_move[1])

                val, response_path = self.min_value(depth - 1, alpha, beta, debug_path)

            finally:
                self.game.undo_move()
                if debug_path is not None:
                    debug_path.pop()

            if val > value:
                value, move = val, next_move
                best_path_for_this_node = [move] + (response_path or [])

            alpha = max(alpha, val)
            if val >= beta:
                # self.store_tt(self.game.hash_key, depth, value, alpha, beta, move) # 存储逻辑可以更复杂
                return value, best_path_for_this_node

        # self.store_tt(self.game.hash_key, depth, value, alpha, beta, move)
        return value, best_path_for_this_node

    def min_value(
        self, depth: int, alpha: float, beta: float, debug_path: Optional[List] = None
    ) -> Tuple[int, Optional[List[Tuple[int, int]]]]:
        """Min value函数，返回 (评估分, 导致该分数的路径)。"""
        if time.time() - self.start_time > self.time_limit:
            raise TimeUpException()

        if self.game.game_over or depth == 0:
            score = GomokuEvalOptimized.evaluate(self.game)
            if debug_path is not None:
                print(f"\n  [LEAF NODE] Path: {debug_path}")
                print(f"  >>> Leads to Score: {score}")
                self.game.print_board()
            return score, []

        ttv, tt_path = self.probe_tt(self.game.hash_key, depth, alpha, beta)
        if ttv is not None:
            return ttv, tt_path

        value, move = float("inf"), None
        best_path_for_this_node = []
        ordered = GomokuEval.generate_sorted_moves2(self.game)

        for next_move in ordered:
            response_path = []
            try:
                if debug_path is not None:
                    debug_path.append(next_move)

                assert self.game.make_move(next_move[0], next_move[1])

                val, response_path = self.max_value(depth - 1, alpha, beta, debug_path)

            finally:
                self.game.undo_move()
                if debug_path is not None:
                    debug_path.pop()

            if val < value:
                value, move = val, next_move
                best_path_for_this_node = [move] + (response_path or [])

            beta = min(beta, val)
            if val <= alpha:
                # self.store_tt(...)
                return value, best_path_for_this_node

        # self.store_tt(...)
        return value, best_path_for_this_node

    def minmax(self, depth: int, game: Gomoku) -> Tuple[int, Optional[Tuple[int, int]]]:
        """Minimax算法，适配返回路径的搜索函数。"""
        self.game = game
        best_move = None
        val = None
        try:
            # max_value 现在返回 val 和 best_path
            val, best_path = self.max_value(depth, float("-inf"), float("inf"))
            # 我们只需要路径中的第一步作为最佳走法
            if best_path:
                best_move = best_path[0]
        except TimeUpException:
            print(f"minmax search timeout at depth {depth}")
            raise TimeUpException()

        return val, best_move

    def iterative_deepening(self, max_depth: int, game: Gomoku) -> Tuple:
        """简单迭代加深：从1到max_depth逐层加深"""
        self.game = game
        self.start_time = time.time()
        self.tt.new_search()

        best_move = None
        best_val = float("-inf")

        for d in range(2, max_depth + 1, 2):
            print(f"searching depth: {d}")
            try:
                # 直接从minmax接收val和move
                val, move = self.minmax(d, self.game)
                if val is not None:
                    best_val = val
                    best_move = move
            except TimeUpException:
                print(f"searching depth: {d}, time up, use depth {d - 2} result")
                break  # 超时后，best_move 自动保留了上一层的有效结果

        # 如果循环因为超时而中断，best_move就是最后一个成功深度的结果
        # 如果循环正常结束，best_move就是max_depth的结果
        print(f"best move: {best_move}, value: {best_val}")

        # 兜底逻辑：如果ID在任何深度都没有找到走法（例如时间设置过短）
        if best_move is None:
            print("ID couldn't find a move, running a shallow search as fallback.")
            _, best_move = self.minmax(2, self.game)  # 运行一个不会超时的浅层搜索

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
            return None, None
        if e.flag == TranspositionTable.EXACT:
            return e.value, e.best_move
        if e.flag == TranspositionTable.LOWER and e.value >= beta:
            return e.value, e.best_move
        if e.flag == TranspositionTable.UPPER and e.value <= alpha:
            return e.value, e.best_move
        return None, None

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

    def debug_move_evaluation(
        self, game: Gomoku, depth: int, move_to_debug: Tuple[int, int]
    ) -> int:
        """
        对特定的一步棋进行带详细日志的评估，找出导致其得分的“最坏路径”。
        """
        print(f"\n========================================================")
        print(f"DEBUGGING MOVE: {move_to_debug} at SEARCH DEPTH: {depth}")
        print(f"========================================================")

        self.game = game
        self.start_time = time.time()
        self.tt.new_search()

        # 为了调试，我们先手动走这步棋
        try:
            assert self.game.make_move(move_to_debug[0], move_to_debug[1])

            # 然后从对手的角度开始搜索，并传入初始路径
            initial_path = [move_to_debug]

            # 注意：因为我们是Max方，走了第一步，所以接下来应该轮到Min方
            # 我们调用min_value来找出对手的最佳应对，从而得到我们这一步的最终得分
            score, best_response_path = self.min_value(
                depth - 1, float("-inf"), float("inf"), debug_path=initial_path
            )

        finally:
            self.game.undo_move()  # 确保在任何情况下都恢复棋盘

        print(f"\n--- DEBUG SUMMARY FOR MOVE {move_to_debug} ---")
        print(f"Final calculated score for this move is: {score}")
        print(f"This score is based on the predicted path: {best_response_path}")
        print(f"========================================================")
        return score
