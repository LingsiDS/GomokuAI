import pygame
import sys
import random
import numpy as np

# ==================== 1. 游戏设置与常量 ====================
# 窗口设置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
BOARD_SIZE = 15
CELL_SIZE = 50
MARGIN = (WINDOW_WIDTH - (BOARD_SIZE - 1) * CELL_SIZE) // 2

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BOARD_COLOR = (194, 178, 128)
HIGHLIGHT_COLOR = (255, 0, 0, 100)
TIP_COLOR_BLACK = (0, 0, 0, 100)
TIP_COLOR_WHITE = (255, 255, 255, 100)

# 棋子颜色
BLACK_STONE = BLACK
WHITE_STONE = WHITE

# Pygame初始化
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("五子棋")

try:
    font = pygame.font.Font("fonts\STKAITI.TTF", 40)
except FileNotFoundError:
    font = pygame.font.Font(None, 40)

running = True


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
        assert self.is_valid_move(row, col), f"({row}, {col}) is invalid position!"
        self.board[row][col] = 0

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


# ==================== 3. AI逻辑 ====================
class AI:
    def __init__(self):
        pass

    def get_random_move(self, board):
        empty_cells = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == 0:
                    empty_cells.append((r, c))
        if empty_cells:
            return random.choice(empty_cells)
        return None


# ==================== 4. Pygame界面绘制 ====================
def draw_board(screen, board, last_move, mouse_pos, current_player):
    screen.fill(BOARD_COLOR)
    for i in range(BOARD_SIZE):
        start_pos_x = MARGIN + i * CELL_SIZE
        start_pos_y = MARGIN + i * CELL_SIZE
        pygame.draw.line(
            screen,
            BLACK,
            (MARGIN, start_pos_y),
            (WINDOW_WIDTH - MARGIN, start_pos_y),
            2,
        )
        pygame.draw.line(
            screen,
            BLACK,
            (start_pos_x, MARGIN),
            (start_pos_x, WINDOW_HEIGHT - MARGIN),
            2,
        )

    center = (BOARD_SIZE // 2, BOARD_SIZE // 2)
    stars = [(3, 3), (3, 11), (11, 3), (11, 11)]
    star_points = stars + [center]
    for r, c in star_points:
        center_pos = (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)
        pygame.draw.circle(screen, BLACK, center_pos, 5)

    if mouse_pos:
        col = round((mouse_pos[0] - MARGIN) / CELL_SIZE)
        row = round((mouse_pos[1] - MARGIN) / CELL_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == 0:
            tip_color = TIP_COLOR_BLACK if current_player == 1 else TIP_COLOR_WHITE
            tip_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(
                tip_surface,
                tip_color,
                (CELL_SIZE // 2, CELL_SIZE // 2),
                CELL_SIZE // 2 - 2,
            )
            screen.blit(
                tip_surface,
                (
                    MARGIN + col * CELL_SIZE - CELL_SIZE // 2,
                    MARGIN + row * CELL_SIZE - CELL_SIZE // 2,
                ),
            )

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == 1:
                color = BLACK_STONE
            elif board[r][c] == 2:
                color = WHITE_STONE
            else:
                continue
            center_pos = (MARGIN + c * CELL_SIZE, MARGIN + r * CELL_SIZE)

            if (r, c) == last_move:
                pygame.draw.circle(screen, HIGHLIGHT_COLOR[:3], center_pos, 5)

            pygame.draw.circle(screen, color, center_pos, CELL_SIZE // 2 - 2)


def draw_text(text, position):
    text_surface = font.render(text, True, BLACK)
    screen.blit(text_surface, position)


def show_game_menu():
    screen.fill(BOARD_COLOR)
    draw_text("五子棋", (WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT // 4))
    draw_text("1. 双人对战", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 50))
    draw_text("2. AI对战", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 50))
    pygame.display.flip()


# ==================== 5. 主循环与事件处理 ====================
def main():
    global running
    game = Gomoku()
    ai = AI()
    game_mode = 0
    mouse_pos = None

    while running:
        # 优化：主循环中统一处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 在菜单模式下，只响应键盘输入
            if game_mode == 0:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        game_mode = 1
                        game.reset_game()
                    elif event.key == pygame.K_2:
                        game_mode = 2
                        game.reset_game()

            # 在游戏模式下，响应鼠标和键盘输入
            else:
                if event.type == pygame.MOUSEMOTION:
                    mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                    if game_mode == 1 or (game_mode == 2 and game.current_player == 1):
                        pos = pygame.mouse.get_pos()
                        col = round((pos[0] - MARGIN) / CELL_SIZE)
                        row = round((pos[1] - MARGIN) / CELL_SIZE)
                        game.make_move(row, col)
                elif event.type == pygame.KEYDOWN and game.game_over:
                    game.reset_game()
                    game_mode = 0

        if game_mode == 0:
            show_game_menu()
        else:
            if game_mode == 2 and game.current_player == 2 and not game.game_over:
                row, col = ai.get_random_move(game.board)
                game.make_move(row, col)

            draw_board(
                screen, game.board, game.last_move, mouse_pos, game.current_player
            )

            if game.game_over:
                winner_text = f"恭喜 {'黑棋' if game.winner == 1 else '白棋'} 获胜！"
                draw_text(
                    winner_text, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 20)
                )
                draw_text(
                    "按任意键返回主菜单",
                    (WINDOW_WIDTH // 2 - 180, WINDOW_HEIGHT // 2 + 30),
                )
            else:
                current_player_text = (
                    f"当前玩家: {'黑棋' if game.current_player == 1 else '白棋'}"
                )
                draw_text(current_player_text, (10, 10))

            pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
