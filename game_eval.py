import re
from gomoku import Gomoku
import numpy as np
from enum import IntEnum
from typing import List
import queue
from collections import defaultdict


class BoardPattern(IntEnum):
    DUMMY = 0
    FIVE = 1
    OPEN_FOUR = 2
    BLOCKED_FOUR = 3
    OPEN_THREE = 4
    BLOCKED_THREE = 5
    OPEN_TWO = 6
    DOUBLE_OPEN_THREE = 7
    DOUBLE_BLOCKED_FOUR = 8
    FOUR_THREE = 9  # 四三杀,一条线是活四，另一条是活三
    DOUBLE_OPEN_FOUR = 10  # 双活四


def pattern2str(pattern: IntEnum):
    dic = {
        1: "FIVE",
        2: "OPEN_FOUR",
        3: "BLOCKED_FOUR",
        4: "OPEN_THREE",
        5: "BLOCKED_THREE",
        6: "OPEN_TWO",
        7: "DOUBLE_OPEN_THREE",
        8: "DOUBLE_BLOCKED_FOUR",
        9: "FOUR_THREE",
        10: "DOUBLE_OPEN_FOUR",
    }
    return dic[pattern]


# TODO: pattern需要细化，以及得分需要细化，同时应该考虑防守
black_patterns = {
    # BoardPattern.DUMMY: [""],
    BoardPattern.FIVE: [re.compile("11111")],
    BoardPattern.OPEN_FOUR: [re.compile("011110")],  # 活四：己方下一手必成五连
    BoardPattern.BLOCKED_FOUR: [
        re.compile("01111(2|$)"),
        re.compile("(2|^)11110"),
        re.compile("0101110"),
        re.compile("0110110"),
        re.compile("0111010"),
    ],  # 冲四的定义是：己方下一步可形成五连，但对手也只需一步可以立即阻止五连形成
    BoardPattern.OPEN_THREE: [
        re.compile("01110"),
        re.compile("010110"),
        re.compile("011010"),
    ],
    BoardPattern.BLOCKED_THREE: [
        re.compile("(2|^)1110"),
        re.compile("0111(2|$)"),
        re.compile("(2|^)10110"),
        re.compile("1101(2|$)"),
    ],
    BoardPattern.OPEN_TWO: [re.compile("0110")],
}

white_patterns = {
    # BoardPattern.DUMMY: [""],
    BoardPattern.FIVE: [re.compile("22222")],  # 白棋五连
    BoardPattern.OPEN_FOUR: [
        re.compile("022220")
    ],  # 白棋活四：两端有空位，下一手必成五
    BoardPattern.BLOCKED_FOUR: [
        re.compile("02222(1|$)"),  # 左侧空位，右侧被黑棋或边界阻挡
        re.compile("(1|^)22220"),  # 左侧被黑棋或边界阻挡，右侧空位
        re.compile("0202220"),  # 间隔型冲四（20222）
        re.compile("0220220"),  # 间隔型冲四（22022）
        re.compile("0222020"),  # 间隔型冲四（22202）
    ],  # 白棋冲四：可成五但被黑棋一步阻挡
    BoardPattern.OPEN_THREE: [
        re.compile("02220"),
        re.compile("020220"),
        re.compile("022020"),
    ],  # 白棋活三
    BoardPattern.BLOCKED_THREE: [
        re.compile("(1|^)2220"),
        re.compile("0222(1|$)"),
        re.compile("(1|^)20220"),
        re.compile("2202(1|$)"),
    ],  # 白棋冲三
    BoardPattern.OPEN_TWO: [re.compile("0220")],  # 白棋活二
}


scores = {
    BoardPattern.FIVE: 8e9,
    BoardPattern.OPEN_FOUR: 20000,
    BoardPattern.BLOCKED_FOUR: 5000,
    BoardPattern.OPEN_THREE: 3000,
    BoardPattern.BLOCKED_THREE: 100,
    BoardPattern.OPEN_TWO: 300,
    BoardPattern.DOUBLE_OPEN_THREE: 15000,  # 分数仅次于活四，优先活四
    BoardPattern.DOUBLE_BLOCKED_FOUR: 20000,  # 基本必杀局
    BoardPattern.FOUR_THREE: 25000,  # 四三杀
    BoardPattern.DOUBLE_OPEN_FOUR: 50000,  # 双活四
}


class GomokuEval:
    @staticmethod
    def get_board_lines(board: np.ndarray) -> List[str]:
        rows = [row for row in board]
        cols = [board[:, i] for i in range(board.shape[1])]

        # from left-up to right-bottom
        diags = [board.diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]

        # from right-up to left-bottom
        anti_diags = [
            np.fliplr(board).diagonal(i)
            for i in range(-board.shape[0] + 1, board.shape[1])
        ]

        lines = rows + cols + diags + anti_diags
        lines = ["".join(x.astype(str)) for x in lines]
        return lines

    @staticmethod
    def evaluate(game: Gomoku):
        """Evaluate the game state and return a score."""
        # print(f"eval: {game.current_player}")
        state_strs = GomokuEval.get_board_lines(game.board)
        own_patterns, opp_patterns = (
            (black_patterns, white_patterns)
            if game.current_player == 1
            else (white_patterns, black_patterns)
        )
        # own_patterns, opp_patterns = (white_patterns, black_patterns)
        game_score = 0
        for state in state_strs:
            for pattern, regexs in own_patterns.items():
                for regex in regexs:
                    if regex.search(state):  # TODO: 多次加分
                        game_score += scores[pattern]
            for pattern, regexs in opp_patterns.items():
                for regex in regexs:
                    if regex.search(state):
                        game_score -= (
                            scores[pattern] * 2.1
                        )  # 对手的棋形权重增加，让AI注重防守
        return game_score

    @staticmethod
    def generate_moves(game: Gomoku):
        """Generate moves and its scores for the current game state."""
        # 先不考虑着法的好坏，生成已有棋子的领域3x3内的空白位置作为候选着法
        # TODO：按着法能形成的棋形进行打分，得分高的候选落子应该具有较高的优先级
        moves = []
        BOARD_SIZE = 15
        x_direc = [-1, 1, 0, 0, -1, -1, 1, 1]  # 上 下 左 右 左上 右上 左下 右下
        y_direc = [0, 0, -1, 1, -1, 1, -1, 1]  # 上 下 左 右 左上 右上 左下 右下

        # first version: BFS
        Q = queue.Queue(225)
        for i in range(15):
            for j in range(15):
                if game.board[i][j] != 0:
                    Q.put((i, j))
        visited = np.zeros((15, 15))
        while not Q.empty():
            t = Q.get(block=False)
            if visited[t[0]][t[1]] == 1:
                continue
            visited[t[0]][t[1]] = 1
            for x, y in zip(x_direc, y_direc):
                nx, ny = t[0] + x, t[1] + y
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    visited[nx][ny] = 1
                    if game.board[nx][ny] == 0:
                        moves.append((nx, ny))
        return moves

    @staticmethod
    def generate_sorted_moves(game: Gomoku):
        moves = []
        BOARD_SIZE = 15
        candidates = []
        # for i in range(BOARD_SIZE):
        #     for j in range(BOARD_SIZE):
        #         if game.board[i][j] == 0:
        #             candidates.append((i, j))
        candidates = GomokuEval.generate_moves(game)

        def is_validate(x, y):
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                return True
            return False

        def check_pattern(x, y, dx, dy, current_player):
            pattern = defaultdict(int)
            nx, ny = x + dx, y + dy
            left_cnt, right_cnt = 0, 0
            is_left_open, is_right_open = False, False

            while is_validate(nx, ny) and game.board[nx][ny] == current_player:
                nx, ny = nx + dx, ny + dy
                left_cnt += 1
            is_left_open = is_validate(nx, ny) and game.board[nx][ny] == 0

            nx, ny = x - dx, y - dy
            while is_validate(nx, ny) and game.board[nx][ny] == current_player:
                nx, ny = nx - dx, ny - dy
                right_cnt += 1
            is_right_open = is_validate(nx, ny) and game.board[nx][ny] == 0

            # check pattern, calc score
            if left_cnt + right_cnt + 1 >= 5:
                pattern[BoardPattern.FIVE] += 1
            elif left_cnt + right_cnt + 1 == 4 and is_left_open and is_right_open:
                pattern[BoardPattern.OPEN_FOUR] += 1
            elif (
                left_cnt + right_cnt + 1 == 4
                and (is_left_open or is_right_open)
                and (left_cnt == 0 or right_cnt == 0)
            ):  # 单边开，且一边为0，冲四
                pattern[BoardPattern.BLOCKED_FOUR] += 1
            elif left_cnt + right_cnt + 1 == 3 and is_left_open and is_right_open:
                pattern[BoardPattern.OPEN_THREE] += 1
            elif left_cnt + right_cnt + 1 == 3 and (is_left_open or is_right_open):
                pattern[BoardPattern.BLOCKED_THREE] += 1
            elif left_cnt + right_cnt + 1 == 2 and is_left_open and is_right_open:
                pattern[BoardPattern.OPEN_TWO] += 1

            return pattern

        def calc_scores(pattern: defaultdict):
            score = 0
            n = len(pattern)
            if n == 1:  # 只有一种模式，可以是单个模式，也可以是双活甚至多活
                for key, value in pattern.items():
                    if value == 1 and n == 1:  # 单个模式的分数，直接加
                        score += scores[key]
                    elif value > 1:  # TODO：双活甚至多活，需要精细优化
                        if key == BoardPattern.OPEN_THREE and value >= 2:
                            score += scores[BoardPattern.DOUBLE_OPEN_THREE]  # 双活三
                            if value > 2:
                                print(
                                    "warning: more than double open three, but not implemented"
                                )
                        elif key == BoardPattern.DOUBLE_BLOCKED_FOUR and value >= 2:
                            score += scores[BoardPattern.DOUBLE_BLOCKED_FOUR]  # 双冲四
                            if value > 2:
                                print(
                                    "warning: more than double blocked four, but not implemented"
                                )
                        elif key == BoardPattern.DOUBLE_OPEN_FOUR and value >= 2:
                            score += scores[BoardPattern.DOUBLE_OPEN_FOUR]  # 双活四
                            if value > 2:
                                print(
                                    "warning: more than double open four, but not implemented"
                                )
            else:  # 多种单个模式
                if BoardPattern.OPEN_THREE in pattern and (
                    BoardPattern.BLOCKED_FOUR in pattern
                    or BoardPattern.OPEN_FOUR in pattern
                ):
                    score += scores[BoardPattern.DOUBLE_OPEN_THREE]
                # TODO: 活三+眠三、冲四+活四
                # elif ...
            return score

        x_direc = [-1, 0, -1, -1]  # 上 左 左上 右上
        y_direc = [0, -1, -1, 1]  # 上 左 左上 右上
        for x, y in candidates:
            score = 0
            own_pattern = defaultdict(int)
            opp_pattern = defaultdict(int)
            for dx, dy in zip(x_direc, y_direc):
                pattern1 = check_pattern(x, y, dx, dy, game.current_player)
                for key, value in pattern1.items():
                    own_pattern[key] += value

                pattern2 = check_pattern(x, y, dx, dy, 3 - game.current_player)
                for key, value in pattern2.items():
                    opp_pattern[key] += value

            own_score = calc_scores(own_pattern)
            opp_score = calc_scores(opp_pattern)
            score = own_score + opp_score
            moves.append((x, y, score))
        moves.sort(key=lambda x: x[2], reverse=True)
        moves = [(x, y) for x, y, _ in moves]  # 只保留坐标，不保留分数
        # print(f"generate_sorted_moves, moves number {len(moves)}")
        return moves
