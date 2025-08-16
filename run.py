import pygame
import sys
import random
import numpy as np
from gomoku import Gomoku
from alpha_beta_search import MinmaxSearch
import threading
import copy
import json, datetime

# ==================== 1. 游戏设置与常量 ====================
# 窗口设置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 900  # 增加窗口高度，为按钮留出空间
BOARD_SIZE = 15
CELL_SIZE = 50
MARGIN = (WINDOW_WIDTH - (BOARD_SIZE - 1) * CELL_SIZE) // 2
BOARD_BOTTOM = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE  # 棋盘底部位置

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BOARD_COLOR = (194, 178, 128)
HIGHLIGHT_COLOR = (255, 0, 0, 100)
TIP_COLOR_BLACK = (0, 0, 0, 100)
TIP_COLOR_WHITE = (255, 255, 255, 100)
BUTTON_COLOR = (100, 150, 200)
BUTTON_HOVER_COLOR = (120, 170, 220)
BUTTON_TEXT_COLOR = WHITE

# 棋子颜色
BLACK_STONE = BLACK
WHITE_STONE = WHITE

# Pygame初始化
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("五子棋")

try:
    font = pygame.font.Font("fonts/STKAITI.TTF", 40)
except FileNotFoundError:
    font = pygame.font.Font(None, 40)

running = True


# ==================== 3. 按钮类 ====================
class Button:
    def __init__(self, x, y, width, height, text, font_size=30):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        # 使用系统默认字体，避免乱码
        try:
            self.font = pygame.font.Font("fonts/STKAITI.TTF", font_size)
        except FileNotFoundError:
            self.font = pygame.font.Font(None, font_size)
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER_COLOR
        self.text_color = BUTTON_TEXT_COLOR
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


# ==================== 4. AI逻辑 ====================
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


# ==================== 5. Pygame界面绘制 ====================
def draw_board(screen, board, last_move, mouse_pos, current_player, buttons=None):
    screen.fill(BOARD_COLOR)
    for i in range(BOARD_SIZE):
        start_pos_x = MARGIN + i * CELL_SIZE
        start_pos_y = MARGIN + i * CELL_SIZE
        pygame.draw.line(
            screen,
            BLACK,
            (MARGIN, start_pos_y),
            (BOARD_BOTTOM, start_pos_y),
            2,
        )
        pygame.draw.line(
            screen,
            BLACK,
            (start_pos_x, MARGIN),
            (start_pos_x, BOARD_BOTTOM),
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

    # 绘制按钮
    if buttons:
        for button in buttons:
            button.draw(screen)


def draw_text(text, position):
    text_surface = font.render(text, True, BLACK)
    screen.blit(text_surface, position)


def show_game_menu():
    screen.fill(BOARD_COLOR)
    draw_text("五子棋", (WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT // 4))
    draw_text("1. 双人对战", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 50))
    draw_text("2. AI对战", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 50))
    pygame.display.flip()


# ==================== 6. 主循环与事件处理 ====================
def main():
    global running
    game = Gomoku()
    ai = MinmaxSearch()
    game_mode = 0
    mouse_pos = None

    ai_move_result = None
    ai_thread = None

    # 创建按钮
    button_width = 120
    button_height = 40
    button_y = BOARD_BOTTOM + 30  # 按钮位于棋盘下方30像素处
    undo_button = Button(
        WINDOW_WIDTH // 2 - button_width - 20,
        button_y,
        button_width,
        button_height,
        "悔棋",
    )
    save_button = Button(
        WINDOW_WIDTH // 2 + 20, button_y, button_width, button_height, "保存棋局"
    )
    buttons = [undo_button, save_button]

    def ai_worker(game_copy):
        nonlocal ai_move_result
        ai_move_result = ai.minmax(depth=4, game=game_copy)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game_mode == 0:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        game_mode = 1
                        game.reset_game()
                    elif event.key == pygame.K_2:
                        game_mode = 2
                        game.reset_game()
            else:
                if event.type == pygame.MOUSEMOTION:
                    mouse_pos = pygame.mouse.get_pos()
                    # 处理按钮悬停
                    for button in buttons:
                        button.handle_event(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # 先处理按钮点击
                    button_clicked = False
                    if undo_button.handle_event(event):
                        button_clicked = True
                        if game_mode == 1:  # 双人对战模式
                            if game.undo_move():
                                print("悔棋完成")
                            else:
                                print("没有可撤销的步骤")
                        elif game_mode == 2:  # AI模式
                            # 如果AI正在思考，先停止AI
                            if ai_thread and ai_thread.is_alive():
                                ai_move_result = None
                                print("AI思考已停止")

                            # 撤销AI落子和玩家落子
                            undo_count = 0
                            if game.undo_move():  # 撤销AI落子
                                undo_count += 1
                            if game.undo_move():  # 撤销玩家落子
                                undo_count += 1

                            if undo_count > 0:
                                print(f"悔棋完成，撤销了{undo_count}步")
                            else:
                                print("没有可撤销的步骤")

                    elif save_button.handle_event(event):
                        button_clicked = True
                        try:
                            snapshot = {
                                "current_player": game.current_player,
                                "board": game.board.tolist(),
                                "history": game.history,
                                "game_mode": game_mode,
                                "game_over": game.game_over,
                                "winner": game.winner,
                                "save_time": datetime.datetime.now().isoformat(),
                                "total_moves": len(game.history),
                            }

                            filename = f"gomoku_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            with open(filename, "w", encoding="utf-8") as f:
                                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                            print(f"已保存棋局快照到 {filename}")
                            print(f"当前步数: {len(game.history)}")
                        except Exception as e:
                            print(f"保存快照失败: {e}")

                    # 如果没有点击按钮，检查是否点击了棋盘
                    if not button_clicked and not game.game_over:
                        pos = pygame.mouse.get_pos()
                        col = round((pos[0] - MARGIN) / CELL_SIZE)
                        row = round((pos[1] - MARGIN) / CELL_SIZE)
                        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                            if game_mode == 1 or (
                                game_mode == 2 and game.current_player == 1
                            ):
                                game.make_move(row, col)

                elif event.type == pygame.KEYDOWN and game.game_over:
                    game.reset_game()
                    game_mode = 0

        if game_mode == 0:
            show_game_menu()
        else:
            # AI线程启动与结果处理
            if game_mode == 2 and game.current_player == 2 and not game.game_over:
                if ai_thread is None or not ai_thread.is_alive():
                    if ai_move_result is None:
                        game_copy = copy.deepcopy(game)
                        ai_thread = threading.Thread(
                            target=ai_worker, args=(game_copy,), daemon=True
                        )
                        ai_thread.start()
                if ai_move_result is not None:
                    row, col = ai_move_result
                    game.make_move(row, col)
                    ai_move_result = None

            draw_board(
                screen,
                game.board,
                game.last_move,
                mouse_pos,
                game.current_player,
                buttons,
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

                # AI思考中提示
                if (
                    game_mode == 2
                    and game.current_player == 2
                    and ai_thread is not None
                    and ai_thread.is_alive()
                ):
                    draw_text("AI思考中...", (WINDOW_WIDTH - 200, WINDOW_HEIGHT - 50))
            pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
