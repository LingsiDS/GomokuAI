import re
from gomoku import Gomoku
import numpy as np
from enum import IntEnum
from typing import List
import queue


class BoardPattern(IntEnum):
    FIVE = 1
    OPEN_FOUR = 2
    BLOCKED_FOUR = 3
    OPEN_THREE = 4
    BLOCKED_THREE = 5
    OPEN_TWO = 6


def pattern2str(pattern: IntEnum):
    dic = {
        1: "FIVE",
        2: "OPEN_FOUR",
        3: "BLOCKED_FOUR",
        4: "OPEN_THREE",
        5: "BLOCKED_THREE",
        6: "OPEN_TWO",
    }
    return dic[pattern]


# TODO: pattern需要细化，以及得分需要细化，同时应该考虑防守
black_patterns = {
    BoardPattern.FIVE: ["11111"],
    BoardPattern.OPEN_FOUR: ["011110"],  # 活四：己方下一手必成五连
    BoardPattern.BLOCKED_FOUR: [
        "01111(2|$)",
        "(2|^)11110",
        "0101110",
        "0110110",
        "0111010",
    ],  # 冲四的定义是：己方下一步可形成五连，但对手也只需一步可以立即阻止五连形成
    BoardPattern.OPEN_THREE: ["01110", "010110", "011010"],
    BoardPattern.BLOCKED_THREE: ["(2|^)1110", "0111(2|$)", "(2|^)10110", "1101(2|$)"],
    BoardPattern.OPEN_TWO: ["0110"],
}

white_patterns = {
    BoardPattern.FIVE: ["22222"],  # 白棋五连
    BoardPattern.OPEN_FOUR: ["022220"],  # 白棋活四：两端有空位，下一手必成五
    BoardPattern.BLOCKED_FOUR: [
        "02222(1|$)",  # 左侧空位，右侧被黑棋或边界阻挡
        "(1|^)22220",  # 左侧被黑棋或边界阻挡，右侧空位
        "0202220",  # 间隔型冲四（20222）
        "0220220",  # 间隔型冲四（22022）
        "0222020",  # 间隔型冲四（22202）
    ],  # 白棋冲四：可成五但被黑棋一步阻挡
    BoardPattern.OPEN_THREE: ["02220", "020220", "022020"],  # 白棋活三
    BoardPattern.BLOCKED_THREE: [
        "(1|^)2220",
        "0222(1|$)",
        "(1|^)20220",
        "2202(1|$)",
    ],  # 白棋冲三
    BoardPattern.OPEN_TWO: ["0220"],  # 白棋活二
}


scores = {
    BoardPattern.FIVE: 8e9,
    BoardPattern.OPEN_FOUR: 20000,
    BoardPattern.BLOCKED_FOUR: 5000,
    BoardPattern.OPEN_THREE: 3000,
    BoardPattern.BLOCKED_THREE: 100,
    BoardPattern.OPEN_TWO: 300,
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
        print(f"eval: {game.current_player}")
        state_strs = GomokuEval.get_board_lines(game.board)
        own_patterns, opp_patterns = (
            (black_patterns, white_patterns)
            if game.current_player == 1
            else (white_patterns, black_patterns)
        )
        own_patterns, opp_patterns = (white_patterns, black_patterns)
        game_score = 0
        for state in state_strs:
            for pattern, regexs in own_patterns.items():
                for regex in regexs:
                    if re.search(regex, state):  # TODO: 多次加分
                        game_score += scores[pattern]
            for pattern, regexs in opp_patterns.items():
                for regex in regexs:
                    if re.search(regex, state):
                        game_score -= scores[pattern]
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
