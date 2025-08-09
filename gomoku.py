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

    def reset_game(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None

    def is_valid_move(self, row, col):
        return (
            0 <= row < BOARD_SIZE
            and 0 <= col < BOARD_SIZE
            and self.board[row][col] == 0
        )

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            self.last_move = (row, col)
            if self.check_win(row, col):
                self.game_over = True
                self.winner = self.current_player
            else:
                self.current_player = 3 - self.current_player
            return True
        return False

    def undo_move(self, row: int, col: int):
        assert (
            0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE
        ), f"({row}, {col}) is invalid position!"
        self.board[row][col] = 0
        self.current_player = 3 - self.current_player

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
