import re
from gomoku import Gomoku
import numpy as np
from enum import IntEnum
from typing import List
import queue
from collections import defaultdict
import pdb


class BoardPattern(IntEnum):
    DUMMY = 0
    FIVE = 1
    OPEN_FOUR = 2
    BLOCKED_FOUR = 3
    OPEN_THREE = 4
    BLOCKED_THREE = 5
    OPEN_TWO = 6
    BLOCKED_TWO = 7
    DOUBLE_OPEN_THREE = 8
    DOUBLE_BLOCKED_FOUR = 9
    FOUR_THREE = 10  # 四三杀,一条线是活四，另一条是活三
    DOUBLE_OPEN_FOUR = 11


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


# 基础分数映射
SCORE_MAP = {
    "FIVE": 8e9,
    "OPEN_FOUR": 20000,
    "BLOCKED_FOUR": 5000,
    "OPEN_THREE": 3000,
    "BLOCKED_THREE": 100,
    "OPEN_TWO": 300,
    "DOUBLE_OPEN_THREE": 15000,
    "DOUBLE_OPEN_FOUR": 50000,
}

# 黑棋大正则
BLACK_UNION = re.compile(
    "(?=(?P<FIVE>11111)|"
    "(?P<OPEN_FOUR>011110)|"
    "(?P<BLOCKED_FOUR>01111(2|3)|(2|3)11110|0101110|0110110|0111010)|"
    "(?P<OPEN_THREE>01110|010110|011010|0101010)|"
    "(?P<BLOCKED_THREE>(2|3)1110|0111(2|3)|(2|3)10110|1101(2|3))|"
    "(?P<OPEN_TWO>0110|01010)|"
    "(?P<BLOCKED_TWO>(2|3)110|011(2|3)|(2|3)1010|0101(2|3)|1001))"
)

# 白棋大正则
WHITE_UNION = re.compile(
    "(?=(?P<FIVE>22222)|"
    "(?P<OPEN_FOUR>022220)|"
    "(?P<BLOCKED_FOUR>02222(1|3)|(1|3)22220|0202220|0220220|0222020)|"
    "(?P<OPEN_THREE>02220|020220|022020|0202020)|"
    "(?P<BLOCKED_THREE>(1|3)2220|0222(1|3)|(1|3)20220|02202(1|3))|"
    "(?P<OPEN_TWO>0220|02020)|"
    "(?P<BLOCKED_TWO>(1|3)220|022(1|3)|(1|3)2020|0202(1|3)))"
)


class GomokuEvalOptimized:
    @staticmethod
    def get_board_lines(board: np.ndarray) -> List[str]:
        rows = [row for row in board]
        cols = [board[:, i] for i in range(board.shape[1])]
        diags = [board.diagonal(i) for i in range(-board.shape[0] + 1, board.shape[1])]
        anti_diags = [
            np.fliplr(board).diagonal(i)
            for i in range(-board.shape[0] + 1, board.shape[1])
        ]
        lines = rows + cols + diags + anti_diags
        lines = ["".join(x.astype(str)) for x in lines]
        return lines

    @staticmethod
    def evaluate(game: Gomoku):
        """优化后的评估函数：支持正则快速扫描，并识别双活三/双活四"""
        lines = GomokuEvalOptimized.get_board_lines(game.board)
        lines = ["3" + line + "3" for line in lines]  # 加哨兵简化边界

        own_union = BLACK_UNION if game.current_player == 1 else WHITE_UNION
        opp_union = WHITE_UNION if game.current_player == 1 else BLACK_UNION

        score = 0
        own_open_three = 0
        own_open_four = 0
        opp_open_three = 0
        opp_open_four = 0

        for line in lines:
            # 自己的棋型
            for m in own_union.finditer(line):
                kind = m.lastgroup
                if kind == "FIVE":
                    return SCORE_MAP["FIVE"]
                if kind == "OPEN_THREE":
                    own_open_three += 1
                if kind == "OPEN_FOUR":
                    own_open_four += 1
                score += SCORE_MAP.get(kind, 0)

            # 对手的棋型
            for m in opp_union.finditer(line):
                kind = m.lastgroup
                if kind == "FIVE":
                    return -SCORE_MAP["FIVE"]
                if kind == "OPEN_THREE":
                    opp_open_three += 1
                if kind == "OPEN_FOUR":
                    opp_open_four += 1
                score -= SCORE_MAP.get(kind, 0)

        # 额外处理双活三、双活四
        if own_open_three >= 2:
            score += SCORE_MAP["DOUBLE_OPEN_THREE"]
        if own_open_four >= 2:
            score += SCORE_MAP["DOUBLE_OPEN_FOUR"]
        if opp_open_three >= 2:
            score -= SCORE_MAP["DOUBLE_OPEN_THREE"]
        if opp_open_four >= 2:
            score -= SCORE_MAP["DOUBLE_OPEN_FOUR"]

        return score


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
        moves = set()  # 使用set来避免重复
        BOARD_SIZE = 15
        x_direc = [-1, 1, 0, 0, -1, -1, 1, 1]  # 上 下 左 右 左上 右上 左下 右下
        y_direc = [0, 0, -1, 1, -1, 1, -1, 1]  # 上 下 左 右 左上 右上 左下 右下

        # first version: BFS
        Q = queue.Queue(225)
        for i in range(15):
            for j in range(15):
                if game.board[i][j] != 0:
                    Q.put((i, j))
        while not Q.empty():
            t = Q.get(block=False)
            for x, y in zip(x_direc, y_direc):
                nx, ny = t[0] + x, t[1] + y
                if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    if game.board[nx][ny] == 0:
                        moves.add((nx, ny))  # 使用add而不是append
        return list(moves)  # 转换为list返回

    @staticmethod
    def calc_scores(pattern: defaultdict):
        """
        计算棋形组合的得分

        棋形威胁等级（从高到低）：
        1. 五连 - 直接获胜
        2. 双活四 - 必胜局
        3. 活四 - 必胜局
        4. 四三杀 - 活四+活三
        5. 双冲四 - 强威胁
        6. 双活三 - 强威胁
        7. 冲四+活三 - 强威胁
        8. 单个冲四 - 中等威胁
        9. 单个活三 - 中等威胁
        10. 多个冲三 - 弱威胁
        11. 单个冲三 - 弱威胁
        12. 多个活二 - 基础威胁
        13. 单个活二 - 基础威胁
        """
        score = 0

        # 提取各种棋形的数量
        five_count = pattern.get(BoardPattern.FIVE, 0)
        open_four_count = pattern.get(BoardPattern.OPEN_FOUR, 0)
        blocked_four_count = pattern.get(BoardPattern.BLOCKED_FOUR, 0)
        open_three_count = pattern.get(BoardPattern.OPEN_THREE, 0)
        blocked_three_count = pattern.get(BoardPattern.BLOCKED_THREE, 0)
        open_two_count = pattern.get(BoardPattern.OPEN_TWO, 0)

        # 1. 五连 - 直接获胜，最高优先级
        if five_count > 0:
            return scores[BoardPattern.FIVE]

        # 2. 双活四 - 必胜局，对手无法同时防守两个活四
        if open_four_count >= 2:
            return scores[BoardPattern.DOUBLE_OPEN_FOUR]

        # 3. 活四 - 必胜局，下一手必成五连
        if open_four_count == 1:
            return scores[BoardPattern.OPEN_FOUR]

        # 4. 四三杀 - 活四+活三，对手只能防守一个，另一个必成
        if open_four_count == 1 and open_three_count >= 1:
            return scores[BoardPattern.FOUR_THREE]

        # 5. 双冲四 - 强威胁，需要对手连续防守
        if blocked_four_count >= 2:
            score += scores[BoardPattern.DOUBLE_BLOCKED_FOUR]

        # 6. 双活三 - 强威胁，对手难以同时防守
        if open_three_count >= 2:
            score += scores[BoardPattern.DOUBLE_OPEN_THREE]

        # 7. 冲四+活三 - 强威胁，对手需要优先防守冲四
        if blocked_four_count >= 1 and open_three_count >= 1:
            # 略低于四三杀，因为冲四不是必胜
            score += scores[BoardPattern.FOUR_THREE] * 0.8

        # 8. 单个冲四 - 中等威胁，需要防守
        if blocked_four_count == 1:
            score += scores[BoardPattern.BLOCKED_FOUR]

        # 9. 单个活三 - 中等威胁，需要防守
        if open_three_count == 1:
            score += scores[BoardPattern.OPEN_THREE]

        # 10. 多个冲三 - 弱威胁，但数量多时也有威胁
        if blocked_three_count >= 2:
            # 多个冲三的威胁比单个大，但不是线性增长
            score += scores[BoardPattern.BLOCKED_THREE] * blocked_three_count * 1.5

        # 11. 单个冲三 - 弱威胁
        if blocked_three_count == 1:
            score += scores[BoardPattern.BLOCKED_THREE]

        # 12. 多个活二 - 基础威胁，为后续发展做准备
        if open_two_count >= 2:
            # 多个活二的价值比单个大
            score += scores[BoardPattern.OPEN_TWO] * open_two_count * 1.2

        # 13. 单个活二 - 基础威胁
        if open_two_count == 1:
            score += scores[BoardPattern.OPEN_TWO]

        return score

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

            # 方法1: 检测连续棋形（原有逻辑）
            nx, ny = x + dx, y + dy
            left_cnt, right_cnt = 0, 0
            left_after_gap_cnt, right_after_gap_cnt = 0, 0
            is_left_open, is_right_open = False, False
            is_left_after_gap_open, is_right_after_gap_open = False, False

            while is_validate(nx, ny) and game.board[nx][ny] == current_player:
                nx, ny = nx + dx, ny + dy
                left_cnt += 1
            is_left_open = is_validate(nx, ny) and game.board[nx][ny] == 0

            nx, ny = nx + dx, ny + dy
            while (
                is_left_open
                and is_validate(nx, ny)
                and game.board[nx][ny] == current_player
            ):
                nx, ny = nx + dx, ny + dy
                left_after_gap_cnt += 1
            is_left_after_gap_open = is_validate(nx, ny) and game.board[nx][ny] == 0

            nx, ny = x - dx, y - dy
            while is_validate(nx, ny) and game.board[nx][ny] == current_player:
                nx, ny = nx - dx, ny - dy
                right_cnt += 1
            is_right_open = is_validate(nx, ny) and game.board[nx][ny] == 0

            nx, ny = nx - dx, ny - dy
            while (
                is_right_open
                and is_validate(nx, ny)
                and game.board[nx][ny] == current_player
            ):
                nx, ny = nx - dx, ny - dy
                right_after_gap_cnt += 1
            is_right_after_gap_open = is_validate(nx, ny) and game.board[nx][ny] == 0

            # check pattern, calc score
            if left_cnt + right_cnt + 1 >= 5:
                pattern[BoardPattern.FIVE] += 1
            if left_cnt + right_cnt + 1 == 4 and is_left_open and is_right_open:
                pattern[BoardPattern.OPEN_FOUR] += 1
            elif (
                left_cnt + right_cnt + 1 == 4
                and (is_left_open or is_right_open)
                and (left_cnt == 0 or right_cnt == 0)
            ):  # 单边开，且一边为0，冲四, e.g. 011112, 211110
                pattern[BoardPattern.BLOCKED_FOUR] += 1
            elif left_cnt + right_cnt + 1 == 4 and (is_left_open or is_right_open):
                pattern[BoardPattern.BLOCKED_FOUR] += 1  # e.g. 0101112, 0110112
            if left_cnt + right_cnt + 1 == 3 and is_left_open and is_right_open:
                pattern[BoardPattern.OPEN_THREE] += 1
            elif left_cnt + right_cnt + 1 == 3 and (is_left_open or is_right_open):
                pattern[BoardPattern.BLOCKED_THREE] += 1
            if left_cnt + right_cnt + 1 == 2 and is_left_open and is_right_open:
                pattern[BoardPattern.OPEN_TWO] += 1
            if left_after_gap_cnt >= right_after_gap_cnt and left_after_gap_cnt > 0:
                # print(
                #     f"left_after_gap_cnt:{left_after_gap_cnt}, right_after_gap_cnt:{right_after_gap_cnt}"
                # )
                # print(f"left_cnt:{left_cnt}, right_cnt:{right_cnt}")
                # print(f"x: {x}, y: {y}, dx: {dx}, dy: {dy}")
                cnt = 1 + left_after_gap_cnt + left_cnt
                if is_left_after_gap_open and is_right_open:
                    if cnt == 4:  # e.g. 0101110
                        pattern[BoardPattern.BLOCKED_FOUR] += 1
                    elif cnt == 3:  # e.g. 010110
                        pattern[BoardPattern.OPEN_THREE] += 1
                elif is_left_after_gap_open or is_left_open:  # BLOCK one side
                    if cnt == 4:  # e.g. 0101112, 21011110
                        pattern[BoardPattern.BLOCKED_FOUR] += 1
                    if cnt == 3:  # e.g. 010112
                        pattern[BoardPattern.BLOCKED_THREE] += 1
            elif left_after_gap_cnt < right_after_gap_cnt and right_after_gap_cnt > 0:
                # print(
                #     f"left_after_gap_cnt:{left_after_gap_cnt}, right_after_gap_cnt:{right_after_gap_cnt}"
                # )
                # print(f"left_cnt:{left_cnt}, right_cnt:{right_cnt}")
                # print(f"x: {x}, y: {y}, dx: {dx}, dy: {dy}")
                cnt = 1 + right_after_gap_cnt + right_cnt
                if is_right_after_gap_open and is_left_open:
                    if cnt == 4:  # e.g. 0111010
                        pattern[BoardPattern.BLOCKED_FOUR] += 1
                    elif cnt == 3:  # e.g. 011010
                        pattern[BoardPattern.OPEN_THREE] += 1
                elif is_right_after_gap_open or is_right_open:
                    if cnt == 4:  # e.g. 2111010, 0111012
                        pattern[BoardPattern.BLOCKED_FOUR] += 1
                    if cnt == 3:  # e.g. 211010
                        pattern[BoardPattern.BLOCKED_THREE] += 1
            return pattern

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

            own_score = GomokuEval.calc_scores(own_pattern)
            opp_score = GomokuEval.calc_scores(opp_pattern)
            own_pattern_str = [pattern2str(p) for p in own_pattern]
            opp_pattern_str = [pattern2str(p) for p in opp_pattern]
            # print(f"own_pattern_str: {own_pattern_str}")
            # print(f"opp_pattern_str: {opp_pattern_str}")
            score = own_score + opp_score
            moves.append((x, y, score, own_pattern_str, opp_pattern_str))
        moves.sort(key=lambda x: x[2], reverse=True)
        # for move in moves:
        #     print(move)
        moves = [(x, y) for x, y, _, _, _ in moves]  # 只保留坐标，不保留分数
        # print(f"generate_sorted_moves, moves number {len(moves)}")
        return moves

    @staticmethod
    def generate_sorted_moves2(game: Gomoku):
        # 把大正则组名映射到你的 BoardPattern（只写你当前 union 里存在的组）
        _GROUP_TO_BP = {
            "FIVE": BoardPattern.FIVE,
            "OPEN_FOUR": BoardPattern.OPEN_FOUR,
            "BLOCKED_FOUR": BoardPattern.BLOCKED_FOUR,
            "OPEN_THREE": BoardPattern.OPEN_THREE,
            "BLOCKED_THREE": BoardPattern.BLOCKED_THREE,
            "OPEN_TWO": BoardPattern.OPEN_TWO,
            # 若你后续在 UNION 里加了 BLOCKED_TWO，这里再补上：
            # "BLOCKED_TWO": BoardPattern.BLOCKED_TWO,
        }

        def _extract_window_str(board, x, y, dx, dy, player, radius=4):
            """
            提取以(x,y)为中心、方向(dx,dy)的窗口串（长度 2*radius+1），
            中心位置强制视为 player 落子；边界用 '3'。并在首尾各加一个 '3' 作为哨兵。
            """
            h, w = board.shape
            chars = []
            for k in range(-radius, radius + 1):
                nx, ny = x + k * dx, y + k * dy
                if nx == x and ny == y:
                    chars.append(str(player))  # 模拟本次落子
                elif 0 <= nx < h and 0 <= ny < w:
                    chars.append(str(board[nx, ny]))
                else:
                    chars.append("3")  # 越界视为阻塞
            # 首尾再加哨兵，与你 evaluate 里的做法保持一致（便于使用 (1|3)/(2|3) 的边界正则）
            return "3" + "".join(chars) + "3"

        def _count_patterns_on_line(line_str, center_idx, union_re):
            """
            在单条方向的窗口串上用大正则扫描；只统计覆盖 center_idx 的匹配。
            由于 UNION 使用了 (?=...) 前瞻，match 是零宽；用命中组的实际文本长度来复原区间。
            """
            counts = defaultdict(int)
            seen = set()  # 去重：同一方向上相同(kind,start,end)只记一次
            for m in union_re.finditer(line_str):
                kind = m.lastgroup
                if not kind:
                    continue
                frag = m.group(kind)
                if not frag:
                    continue
                s = m.start()
                e = s + len(frag) - 1
                if s <= center_idx <= e:
                    key = (kind, s, e)
                    if key in seen:
                        continue
                    seen.add(key)
                    bp = _GROUP_TO_BP.get(kind)
                    if bp is not None:
                        counts[bp] += 1
            return counts

        def check_pattern(x, y, dx, dy, current_player):
            """
            方向化的按位检测函数（与原有签名一致）：
            返回一个 defaultdict(BoardPattern -> count)，只统计包含 (x,y) 这个落点的棋形。
            """
            # 取对应的大正则（沿用你文件里的 BLACK_UNION / WHITE_UNION）
            union_re = BLACK_UNION if current_player == 1 else WHITE_UNION

            # 取一条方向窗口串，并计算“中心点”在串中的下标
            # 窗口主体长度 = 2*radius + 1；前面多了 1 个哨兵 '3'，所以中心下标 = 1 + radius
            radius = 4
            line_str = _extract_window_str(
                game.board, x, y, dx, dy, current_player, radius
            )
            center_idx = 1 + radius

            # 只统计覆盖 center_idx 的匹配
            return _count_patterns_on_line(line_str, center_idx, union_re)

        """
        评估在 (x,y) 落子后的得分。
        - 仅考虑该点四个主要方向 (水平、竖直、主对角、副对角)
        - 用大正则 (BLACK_UNION/WHITE_UNION) 检测基础棋型
        - 再调用 calc_scores 得到组合型分数
        返回: (own_score, opp_score, own_patterns, opp_patterns)
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        moves = []
        BOARD_SIZE = 15
        candidates = []
        # for i in range(BOARD_SIZE):
        #     for j in range(BOARD_SIZE):
        #         if game.board[i][j] == 0:
        #             candidates.append((i, j))
        candidates = GomokuEval.generate_moves(game)
        for x, y in candidates:
            own_pattern = defaultdict(int)
            opp_pattern = defaultdict(int)
            for dx, dy in directions:
                # 当前玩家落子后的棋型
                p1 = check_pattern(x, y, dx, dy, game.current_player)
                for k, v in p1.items():
                    own_pattern[k] += v

                # 模拟对手在该点落子（对手威胁）
                p2 = check_pattern(x, y, dx, dy, 3 - game.current_player)
                for k, v in p2.items():
                    opp_pattern[k] += v

            # 转成分数
            own_score = GomokuEval.calc_scores(own_pattern)
            opp_score = GomokuEval.calc_scores(opp_pattern)
            moves.append(
                (
                    x,
                    y,
                    own_score + opp_score,
                    [pattern2str(p) for p in own_pattern],
                    [pattern2str(p) for p in opp_pattern],
                )
            )
        moves.sort(key=lambda x: x[2], reverse=True)
        # print(f"moves: {moves}")
        moves = [(x, y) for x, y, _, _, _ in moves]  # 只保留坐标，不保留分数
        return moves
