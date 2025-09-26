import time
import cProfile
import pstats
import io
from gomoku import Gomoku
from alpha_beta_search import AlphaBetaSearch
from game_eval import GomokuEval, GomokuEvalOptimized
import numpy as np
import json


def create_test_position():
    """创建一个测试棋局"""
    game = Gomoku()

    # 创建一个有威胁的棋局
    board_snapshot = json.load(
        open("board_snapshots/gomoku_snapshot_20250816_183325.json")
    )
    game.board = np.array(board_snapshot["board"])
    game.current_player = board_snapshot["current_player"]
    # test_moves = [
    #     (7, 7),  # 中心开局
    #     (7, 8),  # 黑棋
    #     (8, 7),  # 白棋
    #     (8, 8),  # 黑棋
    #     (6, 6),  # 白棋
    #     (6, 8),  # 黑棋 - 形成威胁
    #     (8, 6),  # 白棋
    #     (9, 7),  # 黑棋
    #     (7, 9),  # 白棋
    #     (9, 8),  # 黑棋
    #     (8, 9),  # 白棋
    #     (9, 9),  # 黑棋
    # ]

    # for i, (row, col) in enumerate(test_moves):
    #     game.make_move(row, col)

    return game


def profile_search_performance():
    """分析搜索性能"""
    print("=== 搜索性能分析 ===")

    game = create_test_position()
    ai = AlphaBetaSearch()

    # 测试不同深度的搜索时间
    depths = [2, 3, 4]
    game.print_board()
    print("hash(game.board.tobytes()):", hash(game.board.tobytes()))
    for depth in depths:
        print(f"\n测试深度 {depth}:")

        # 预热
        game_copy = Gomoku()
        game_copy.board = game.board.copy()
        game_copy.current_player = game.current_player
        game_copy.history = game.history.copy()

        # 实际测试
        start_time = time.time()
        result = ai.minmax(depth, game_copy)
        end_time = time.time()

        print(f"  搜索时间: {end_time - start_time:.3f}秒")
        print(f"  搜索结果: {result}")


def profile_evaluation_performance():
    """分析评估函数性能"""
    print("\n=== 评估函数性能分析 ===")

    game = create_test_position()

    # 测试评估函数性能
    iterations = 1000

    start_time = time.time()
    for _ in range(iterations):
        score = GomokuEval.evaluate(game)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"评估函数平均执行时间: {avg_time*1000:.3f}毫秒")
    print(f"当前局面评分: {score}")

    start_time = time.time()
    for _ in range(iterations):
        opt_score = GomokuEvalOptimized.evaluate(game)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"优化评估函数平均执行时间: {avg_time*1000:.3f}毫秒")
    print(f"当前局面评分: {opt_score}")


def profile_move_generation():
    """分析移动生成性能"""
    print("\n=== 移动生成性能分析 ===")

    game = create_test_position()

    # 测试移动生成性能
    iterations = 50

    start_time = time.time()
    for _ in range(iterations):
        moves = GomokuEval.generate_sorted_moves(game)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    print(f"移动生成平均执行时间: {avg_time*1000:.3f}毫秒")
    print(f"生成移动数量: {len(GomokuEval.generate_sorted_moves(game))}")


def detailed_profiling():
    """详细性能分析"""
    print("\n=== 详细性能分析 ===")

    game = create_test_position()
    ai = AlphaBetaSearch()

    # 使用cProfile进行详细分析
    pr = cProfile.Profile()
    pr.enable()

    # 执行一次完整的AI搜索
    result = ai.minmax(4, game)

    pr.disable()

    # 输出分析结果
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # 显示前20个最耗时的函数

    print("性能分析结果 (按累计时间排序):")
    print(s.getvalue())


def analyze_memory_usage():
    """分析内存使用情况"""
    print("\n=== 内存使用分析 ===")

    import sys
    import gc

    game = create_test_position()
    ai = AlphaBetaSearch()

    # 强制垃圾回收
    gc.collect()

    # 记录初始内存
    initial_memory = sys.getsizeof(game.board) + sys.getsizeof(game.history)

    # 执行搜索
    result = ai.minmax(3, game)

    # 记录搜索后内存
    final_memory = sys.getsizeof(game.board) + sys.getsizeof(game.history)

    print(f"棋盘内存占用: {sys.getsizeof(game.board)} 字节")
    print(f"历史记录内存占用: {sys.getsizeof(game.history)} 字节")
    print(f"内存增长: {final_memory - initial_memory} 字节")


def generate_optimization_suggestions():
    """生成优化建议"""
    print("\n=== 优化建议 ===")

    suggestions = [
        "1. 实现威胁空间搜索，只考虑有威胁的位置",
        "2. 使用位运算替代正则表达式进行棋型识别",
        "3. 实现评估函数缓存，避免重复计算",
        "4. 添加转置表，缓存搜索结果",
        "5. 实现迭代加深搜索",
    ]

    for suggestion in suggestions:
        print(f"  {suggestion}")


def main():
    """主函数"""
    print("五子棋AI性能分析工具")
    print("=" * 50)

    try:
        # 执行各项性能分析
        profile_search_performance()
        profile_evaluation_performance()
        profile_move_generation()
        analyze_memory_usage()
        detailed_profiling()
        generate_optimization_suggestions()

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
