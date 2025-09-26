#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五子棋残局分析工具 (适配最新版本)
用于分析残局、调试AI决策、评估局面威胁等
"""

import json
import numpy as np
import time
import os
from typing import List, Tuple, Optional
from gomoku import Gomoku
from alpha_beta_search import AlphaBetaSearch
from game_eval import GomokuEval, GomokuEvalOptimized
from constants import BOARD_SIZE
import copy


def load_snapshot(filename: str) -> Optional[Gomoku]:
    """加载snapshot文件"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        game = Gomoku()
        game.board = np.array(data["board"])
        game.current_player = data["current_player"]
        game.history = [tuple(move) for move in data["history"]]
        game.game_over = data["game_over"]
        game.winner = data["winner"]
        game.hash_key = data["hash_key"]
        if game.history:
            game.last_move = game.history[-1][:2]

        print(f"✅ 成功加载残局: {os.path.basename(filename)}")
        print(f"   当前玩家: {'黑棋' if game.current_player == 1 else '白棋'}")
        print(f"   历史步数: {len(game.history)}")
        print(f"   游戏状态: {'结束' if game.game_over else '进行中'}")
        if game.winner:
            print(f"   获胜者: {'黑棋' if game.winner == 1 else '白棋'}")

        return game
    except Exception as e:
        print(f"❌ 加载残局失败: {e}")
        return None


def print_board(game: Gomoku):
    """打印棋盘"""
    print("\n当前棋盘状态:")
    print("   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4")
    for i in range(BOARD_SIZE):
        row_str = f"{i:2d}"
        for j in range(BOARD_SIZE):
            if game.board[i][j] == 0:
                row_str += " ."
            elif game.board[i][j] == 1:
                row_str += " ○"
            else:
                row_str += " ●"
        print(row_str)
    print(f"当前玩家: {'黑棋：○' if game.current_player == 1 else '白棋：●'}")


def evaluate_positions_with_minmax(
    game: Gomoku, depth: int = 4, time_limit: float = 10.0
) -> List[Tuple]:
    """使用Minimax搜索评估关键位置的得分"""
    print(f"\n=== 关键位置Minimax得分评估 (深度{depth}, 时间限制{time_limit}s) ===")

    # 使用generate_sorted_moves生成关键位置
    key_positions = GomokuEval.generate_sorted_moves2(game)

    print(f"生成了 {len(key_positions)} 个关键位置")
    print("正在评估每个位置的Minimax得分...")

    positions_scores = []

    # 分析每个位置的Minimax得分
    for idx, (i, j) in enumerate(key_positions):
        print(f"评估进度: {idx+1}/{len(key_positions)} - 位置({i},{j})", end="")

        # 模拟落子
        game.make_move(i, j)

        # 使用Minimax搜索计算得分
        start_time = time.time()
        try:
            # 创建一个临时的AI实例来评估这个位置
            temp_ai = AlphaBetaSearch(time_limit)
            temp_ai.game = game
            temp_ai.start_time = time.time()

            # 模拟落子后，下一步是对方下棋，所以调用min_value
            minmax_score = temp_ai.min_value(depth - 1, float("-inf"), float("inf"))
            print(" minmax_score: ", minmax_score, end="")
            print(" tt: ", temp_ai.tt.get(game.hash_key))
            search_success = True
        except Exception as e:
            print(f"    搜索失败: {e}")
            minmax_score = 0
            search_success = False

        end_time = time.time()
        search_time = end_time - start_time

        # 撤销落子
        game.undo_move()

        if search_success:
            positions_scores.append((i, j, minmax_score, search_time))
        else:
            positions_scores.append((i, j, 0, search_time))

    # 按得分排序
    positions_scores.sort(key=lambda x: x[2], reverse=True)

    print("\n前15个最佳位置 (按Minimax得分排序):")
    print("位置\tMinimax得分\t搜索时间")
    print("-" * 60)

    for i, (row, col, score, search_time) in enumerate(positions_scores[:15]):
        # if score > 1e8:
        #     threat = "🎯 必胜!"
        # elif score > 20000:
        #     threat = "🔥 活四"
        # elif score > 5000:
        #     threat = "⚡ 冲四"
        # elif score > 3000:
        #     threat = "💪 活三"
        # elif score > 100:
        #     threat = "📈 冲三"
        # else:
        #     threat = "📊 普通"

        print(f"({row},{col})\t{score:8.0f}\t\t{search_time:.3f}s")

    return positions_scores


def analyze_ai_decision(
    game: Gomoku, depth: int = 4, time_limit: float = 10.0
) -> Optional[Tuple[int, int]]:
    """分析AI的决策"""
    print(f"\n=== AI决策分析 (深度{depth}, 时间限制{time_limit}s) ===")

    ai = AlphaBetaSearch(time_limit)

    # AI决策
    assert game.current_player == 2, "AI决策时，当前玩家应该是白棋"
    game_copy = copy.deepcopy(game)

    print(f"局面哈希值: {game.hash_key}")
    print_board(game)

    start_time = time.time()
    print(f"开始搜索，深度: {depth}, 时间限制: {time_limit}")
    try:
        ai.minmax(depth, game)
        e = ai.tt.get(game.hash_key)
        ai_move = e.best_move
        search_success = True
    except Exception as e:
        print(f"AI搜索失败: {e}")
        e = ai.tt.get(game.hash_key)
        ai_move = e.best_move if e else None
        search_success = False

    end_time = time.time()

    if search_success and ai_move:
        print(f"AI选择位置: {ai_move}")
        print(f"搜索时间: {end_time - start_time:.3f}秒")

        # 获取关键位置的Minimax得分
        all_scores = evaluate_positions_with_minmax(game_copy, depth, time_limit)
        if all_scores:
            best_position = all_scores[0]

            print(f"\n最佳位置: {best_position[:2]}, Minimax得分: {best_position[2]}")

            if best_position[:2] != ai_move:
                print("⚠️  AI没有选择Minimax得分最高的位置!")
                print(f"AI选择: {ai_move}")
                print(f"最佳选择: {best_position[:2]}, 得分: {best_position[2]}")

                # 找到AI选择的位置在排名中的位置
                ai_rank = None
                for rank, (row, col, score, _) in enumerate(all_scores):
                    if (row, col) == ai_move:
                        ai_rank = rank + 1
                        break

                if ai_rank:
                    print(f"AI选择的位置排名第{ai_rank}位")

                # 分析为什么AI没有选择最佳位置
                print(f"\n=== 问题分析 ===")
                print("可能的原因:")
                print("1. 搜索深度不够，没有看到深层威胁")
                print("2. 评估函数有问题，没有正确识别威胁")
                print("3. 移动生成有问题，遗漏了关键位置")
                print("4. Alpha-Beta剪枝过度，剪掉了最佳分支")
                print("5. 搜索时间限制，没有完成完整搜索")
            else:
                print("✅ AI选择了Minimax得分最高的位置")
    else:
        print("❌ AI搜索失败或没有返回有效走法")

    return ai_move


def analyze_specific_position(
    game: Gomoku, position: Tuple[int, int], depth: int = 6, time_limit: float = 30.0
):
    """专门分析特定位置的详细情况"""
    row, col = position
    print(f"\n=== 深入分析位置 ({row}, {col}) ===")

    # 检查位置是否为空
    if game.board[row][col] != 0:
        print(f"❌ 位置({row},{col})已被占用，无法落子")
        return

    # 分析落子前的局面
    print("落子前的局面评估:")
    original_player = game.current_player

    # 黑棋评估
    game.current_player = 1
    black_score_before = GomokuEvalOptimized.evaluate(game)
    print(f"  黑棋得分: {black_score_before}")

    # 白棋评估
    game.current_player = 2
    white_score_before = GomokuEvalOptimized.evaluate(game)
    print(f"  白棋得分: {white_score_before}")

    # 模拟落子
    print(f"\n模拟在({row},{col})落子..., current_player: {game.current_player}")
    game.make_move(row, col)

    # 分析落子后的局面
    print("落子后的局面评估:")
    print(f"current_player: {game.current_player}")

    # 黑棋评估
    game.current_player = 1
    black_score_after = GomokuEvalOptimized.evaluate(game)
    print(f"  黑棋得分: {black_score_after}")

    # 白棋评估
    game.current_player = 2
    white_score_after = GomokuEvalOptimized.evaluate(game)
    print(f"  白棋得分: {white_score_after}")

    # 计算得分变化
    black_change = black_score_after - black_score_before
    white_change = white_score_after - white_score_before

    print(f"\n得分变化:")
    print(f"  黑棋变化: {black_change:+d}")
    print(f"  白棋变化: {white_change:+d}")

    # 检查是否形成威胁
    print(f"\n威胁分析:")
    if white_change > 20000:
        print(f"  🎯 白棋形成活四威胁!")
    elif white_change > 5000:
        print(f"  ⚡ 白棋形成冲四威胁!")
    elif white_change > 3000:
        print(f"  💪 白棋形成活三威胁!")
    elif white_change > 100:
        print(f"  📈 白棋形成冲三威胁!")
    else:
        print(f"  📊 白棋威胁较低")

    # 使用Minimax搜索评估
    print(f"\nMinimax搜索评估 (深度{depth}):")
    try:
        temp_ai = AlphaBetaSearch(time_limit)
        game.undo_move()
        temp_ai.game = game
        temp_ai.start_time = time.time()

        start_time = time.time()
        minmax_score = temp_ai.max_value(depth, float("-inf"), float("inf"))
        end_time = time.time()

        print(f"  Minmax得分: {minmax_score}")
        print(f"  搜索时间: {end_time - start_time:.3f}s")

        if minmax_score > 1e8:
            print("  🎯 必胜局面!")
        elif minmax_score > 20000:
            print("  🔥 活四威胁!")
        elif minmax_score > 5000:
            print("  ⚡ 冲四威胁!")
        elif minmax_score > 3000:
            print("  💪 活三威胁!")
        elif minmax_score > 100:
            print("  📈 冲三威胁!")
        else:
            print("  📊 普通威胁")

        print("entry: ", temp_ai.tt.get(game.hash_key))

    except Exception as e:
        print(f"  ❌ Minmax搜索失败: {e}")

    # 撤销落子
    game.undo_move()

    # 恢复原始玩家
    game.current_player = original_player

    print(f"\n=== 位置({row},{col})分析完成 ===")


def analyze_threats(game: Gomoku) -> Tuple[float, float]:
    """分析当前局面的威胁"""
    print("\n=== 威胁分析 ===")

    # 保存原始当前玩家
    original_player = game.current_player

    # 分析黑棋威胁
    print("黑棋威胁分析:")
    game.current_player = 1
    black_score = GomokuEvalOptimized.evaluate(game)
    print(f"  黑棋局面得分: {black_score}")

    # 分析白棋威胁
    print("白棋威胁分析:")
    game.current_player = 2
    white_score = GomokuEvalOptimized.evaluate(game)
    print(f"  白棋局面得分: {white_score}")

    # 恢复原始当前玩家
    game.current_player = original_player

    return black_score, white_score


def list_available_snapshots() -> List[str]:
    """列出可用的残局文件"""
    snapshot_dir = "board_snapshots"
    if not os.path.exists(snapshot_dir):
        print(f"❌ 残局目录不存在: {snapshot_dir}")
        return []

    snapshot_files = []
    for file in os.listdir(snapshot_dir):
        if file.endswith(".json"):
            snapshot_files.append(os.path.join(snapshot_dir, file))

    return sorted(snapshot_files)


def select_snapshot() -> Optional[str]:
    """选择要分析的残局文件"""
    snapshot_files = list_available_snapshots()

    if not snapshot_files:
        print("❌ 没有找到可用的残局文件")
        return None

    print("\n可用的残局文件:")
    for i, file_path in enumerate(snapshot_files):
        filename = os.path.basename(file_path)
        print(f"{i+1}. {filename}")

    try:
        choice = input(f"\n请选择要分析的残局 (1-{len(snapshot_files)}): ").strip()
        choice_idx = int(choice) - 1

        if 0 <= choice_idx < len(snapshot_files):
            return snapshot_files[choice_idx]
        else:
            print("❌ 无效的选择")
            return None
    except (ValueError, KeyboardInterrupt):
        print("❌ 输入无效或取消操作")
        return None


def main():
    """主函数"""
    print("五子棋残局分析工具 (最新版本)")
    print("=" * 50)

    # 选择残局文件
    snapshot_file = select_snapshot()
    if not snapshot_file:
        return

    # 加载残局
    game = load_snapshot(snapshot_file)
    if game is None:
        return

    # 打印棋盘
    print_board(game)

    # 分析威胁
    black_score, white_score = analyze_threats(game)

    # 分析AI决策
    ai_move = analyze_ai_decision(game, depth=4, time_limit=64.0)

    # 专门分析(5,9)位置
    print("\n" + "=" * 50)
    analyze_specific_position(game, (5, 7), depth=4, time_limit=64.0)  # 增加搜索深度

    print("\n=== 分析完成 ===")
    print(f"黑棋得分: {black_score}")
    print(f"白棋得分: {white_score}")
    if ai_move:
        print(f"AI推荐走法: {ai_move}")


if __name__ == "__main__":
    main()
