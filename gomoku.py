import numpy as np

from constants import BOARD_SIZE
from zobrist_table import Zobrist


# ==================== 2. 游戏核心逻辑 ====================
class Gomoku:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.history = []  # 存储落子历史
        self.hash_key = 0
        self.zobrist = Zobrist()

    def reset_game(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.history.clear()
        self.hash_key = 0

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
            self.hash_key ^= self.zobrist.piece_key(row, col, self.current_player)
            self.last_move = (row, col)
            self.history.append((row, col, self.current_player))
            if self.check_win(row, col):
                self.game_over = True
                self.winner = self.current_player
            else:  # 只有在游戏没结束时才切换玩家
                self.hash_key ^= self.zobrist.side_key
                self.current_player = 3 - self.current_player
            return True
        return False

    def undo_move(self):
        if not self.history:
            return False

        row, col, player_who_moved = self.history.pop()
        was_game_over = self.game_over
        self.board[row][col] = 0
        self.game_over = False
        self.winner = None

        self.current_player = player_who_moved
        # 恢复哈希值，注意，只有在游戏没有结束的时候恢复
        self.hash_key ^= self.zobrist.piece_key(row, col, player_who_moved)
        if not was_game_over:
            self.hash_key ^= self.zobrist.side_key

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
