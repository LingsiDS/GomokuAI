# debug_ai.py
import numpy as np
import json
from gomoku import Gomoku
from alpha_beta_search import AlphaBetaSearch


# 这是一个简化的 load_snapshot 函数，你需要确保它和你的 run.py 中的一致
def load_snapshot(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        game = Gomoku()
        game.board = np.array(data["board"])
        game.current_player = data["current_player"]
        game.history = [tuple(move) for move in data["history"]]
        game.hash_key = data["hash_key"]
        return game, data.get("game_mode", 2)
    except Exception as e:
        print(f"加载残局失败: {e}")
        return None, None


def main_debug():
    # 1. 加载那个黑棋有活三的局面
    game, _ = load_snapshot("./board_snapshots/gomoku_snapshot_20250907_152307.json")
    if game is None:
        return

    print("--- 初始棋局 ---")
    game.print_board()

    ai = AlphaBetaSearch(time_limit=30)
    debug_depth = 4

    # 2. 调试AI为什么觉得防守点 (5, 7) 不好
    # ai.debug_move_evaluation(game=game, depth=debug_depth, move_to_debug=(5, 7))

    # 3. 调试AI为什么觉得 (5, 9) 更好 (作为对比)
    ai.debug_move_evaluation(game=game, depth=debug_depth, move_to_debug=(5, 9))


if __name__ == "__main__":
    main_debug()
