import json
import numpy as np
from gomoku import Gomoku
from alpha_beta_search import AlphaBetaSearch
from game_eval import GomokuEval, GomokuEvalOptimized
import copy


def test_ai_consistency():
    """测试AI搜索的一致性"""
    print("测试AI搜索一致性")
    print("=" * 50)

    # 加载相同的残局
    with open(
        "board_snapshots/gomoku_snapshot_20250816_183325.json", "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    print(f"残局信息:")
    print(f"  当前玩家: {data['current_player']}")
    print(f"  游戏模式: {data['game_mode']}")
    print(f"  AI悔棋标记: {data.get('ai_undone', False)}")

    # 创建两个完全相同的游戏实例
    game1 = Gomoku()
    game1.board = np.array(data["board"])
    game1.current_player = data["current_player"]
    game1.history = [tuple(move) for move in data["history"]]
    game1.game_over = data["game_over"]
    game1.winner = data["winner"]
    if game1.history:
        game1.last_move = game1.history[-1][:2]

    game2 = copy.deepcopy(game1)

    print(f"\n游戏状态验证:")
    print(f"  game1当前玩家: {game1.current_player}")
    print(f"  game2当前玩家: {game2.current_player}")
    print(f"  状态一致: {game1.current_player == game2.current_player}")

    # 使用相同的AI实例进行搜索
    ai = AlphaBetaSearch()

    print(f"\n=== 第一次搜索 (game1) ===")
    move1 = ai.minmax(depth=4, game=game1)
    print(f"AI选择: {move1}")

    print(f"\n=== 第二次搜索 (game2) ===")
    move2 = ai.minmax(depth=4, game=game2)
    print(f"AI选择: {move2}")

    print(f"\n=== 结果比较 ===")
    print(f"第一次选择: {move1}")
    print(f"第二次选择: {move2}")
    print(f"选择一致: {move1 == move2}")

    if move1 != move2:
        print("⚠️  警告：AI选择不一致！")
        print("可能的原因：")
        print("1. 搜索过程中有随机性")
        print("2. 游戏状态被修改")
        print("3. 评估函数有副作用")
    else:
        print("✅ AI选择一致")


if __name__ == "__main__":
    test_ai_consistency()
