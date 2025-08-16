import numpy as np

BOARD_SIZE = 15


# ==================== 2. 游戏核心逻辑 ====================
class Gomoku:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.history = []  # 存储落子历史

    def reset_game(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.history.clear()

    def is_valid_move(self, row, col):
        if (
            0 <= row < BOARD_SIZE
            and 0 <= col < BOARD_SIZE
            and self.board[row][col] == 0
        ):
            return True
        return False

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.last_move = (row, col)
            self.history.append((row, col, self.current_player))  # 记录历史
            if self.check_win(row, col):
                self.game_over = True
                self.winner = self.current_player
            # 即使赢了也要修改当前玩家，因为在下一次调用evaluate时，对手赢了得分才会变成负数，才能适配minmax搜索
            # 比如黑棋赢了，他继续递归时，他会在max_value的第一行检测到game_over，然后调用evaluate，
            # 这时current_player必须为白棋，才会使得黑棋的得分为负数，从而黑棋棋局的min_value选择这一最优走法，否则会选择其他无关紧要的走法
            # TODO：看看是否能优化逻辑？直接在make_move后面检测是否game_over?
            self.current_player = 3 - self.current_player
            return True
        return False

    def undo_move(self):
        if not self.history:
            print("没有可撤销的步骤")
            return False
        row, col, player = self.history.pop()
        self.board[row][col] = 0
        self.current_player = player
        self.game_over = False
        self.winner = None
        if self.history:
            self.last_move = self.history[-1][:2]
        else:
            self.last_move = None
        return True

    def check_win(self, row, col):
        player = self.board[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if (
                    0 <= r < BOARD_SIZE
                    and 0 <= c < BOARD_SIZE
                    and self.board[r][c] == player
                ):
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if (
                    0 <= r < BOARD_SIZE
                    and 0 <= c < BOARD_SIZE
                    and self.board[r][c] == player
                ):
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def check_last_move_win(self):
        if self.last_move is None:
            return False
        return self.check_win(self.last_move[0], self.last_move[1])

    def print_board(self):
        """打印棋盘"""
        print("当前棋盘状态:")
        print("   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4")
        for i in range(15):
            row_str = f"{i:2d}"
            for j in range(15):
                if self.board[i][j] == 0:
                    row_str += " ."
                elif self.board[i][j] == 1:
                    row_str += " ○"
                else:
                    row_str += " ●"
            print(row_str)
        print(f"当前玩家: {'玩家黑棋：○' if self.current_player == 1 else 'AI白棋：●'}")
