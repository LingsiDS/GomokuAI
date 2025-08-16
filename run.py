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
    small_font = pygame.font.Font("fonts/STKAITI.TTF", 24)
    middle_font = pygame.font.Font("fonts/STKAITI.TTF", 32)
except FileNotFoundError:
    font = pygame.font.Font(None, 40)
    small_font = pygame.font.Font(None, 24)
    middle_font = pygame.font.Font(None, 32)

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
                # 高亮显示最后一步：红色外切矩形边框
                pygame.draw.circle(screen, color, center_pos, CELL_SIZE // 2 - 2)
                # 绘制外切矩形边框
                rect_size = CELL_SIZE - 4
                rect_pos = (
                    center_pos[0] - rect_size // 2,
                    center_pos[1] - rect_size // 2,
                )
                pygame.draw.rect(
                    screen,
                    (255, 0, 0),
                    (rect_pos[0], rect_pos[1], rect_size, rect_size),
                    2,
                )
            else:
                pygame.draw.circle(screen, color, center_pos, CELL_SIZE // 2 - 2)

    # 绘制按钮
    if buttons:
        for button in buttons:
            button.draw(screen)


def draw_text(text, position, color=BLACK, font_type: str = "small"):
    if font_type == "small":
        text_surface = small_font.render(text, True, color)
    elif font_type == "middle":
        text_surface = middle_font.render(text, True, color)
    else:
        text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def show_game_menu():
    screen.fill(BOARD_COLOR)
    draw_text("五子棋", (WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT // 4))
    draw_text("1. 双人对战", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 80))
    draw_text("2. AI对战", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 - 20))
    draw_text("3. 加载残局", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 40))
    pygame.display.flip()


def show_load_snapshot_menu():
    """显示加载残局菜单"""
    screen.fill(BOARD_COLOR)
    draw_text("选择残局文件", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 4))

    # 获取board_snapshots目录下的所有json文件
    import os
    import glob

    snapshot_files = []
    if os.path.exists("board_snapshots"):
        snapshot_files = glob.glob("board_snapshots/*.json")

    if not snapshot_files:
        draw_text("没有找到残局文件", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2))
        draw_text("按ESC返回主菜单", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 50))
    else:
        # 显示前10个文件
        for i, file_path in enumerate(snapshot_files[:10]):
            filename = os.path.basename(file_path)
            draw_text(
                f"{i+1}. {filename}",
                (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 100 + i * 30),
            )

        if len(snapshot_files) > 10:
            draw_text("...", (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 + 200))

        draw_text("按ESC返回主菜单", (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 50))

    pygame.display.flip()
    return snapshot_files


def load_snapshot(filename):
    """加载残局文件"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        game = Gomoku()
        game.board = np.array(data["board"])
        game.current_player = data["current_player"]
        game.history = [tuple(move) for move in data["history"]]
        game.game_over = data["game_over"]
        game.winner = data["winner"]

        if game.history:
            game.last_move = game.history[-1][:2]

        return game, data.get("game_mode", 2)  # 默认AI模式
    except Exception as e:
        print(f"加载残局失败: {e}")
        return None, None


# ==================== 6. 主循环与事件处理 ====================
def main():
    global running
    game = Gomoku()
    ai = MinmaxSearch()
    game_mode = 0  # 0: 主菜单, 1: 双人对战, 2: AI对战, 3: 加载残局菜单
    mouse_pos = None

    ai_move_result = None
    ai_thread = None
    snapshot_files = []  # 存储残局文件列表
    ai_think_start_time = None  # AI开始思考的时间
    ai_think_time = None  # AI思考用时

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
        nonlocal ai_move_result, ai_think_time
        start_time = datetime.datetime.now()
        ai_move_result = ai.minmax(depth=4, game=game_copy)
        end_time = datetime.datetime.now()
        ai_think_time = (end_time - start_time).total_seconds()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game_mode == 0:  # 主菜单
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        game_mode = 1
                        game.reset_game()
                    elif event.key == pygame.K_2:
                        game_mode = 2
                        game.reset_game()
                    elif event.key == pygame.K_3:
                        game_mode = 3  # 进入残局加载菜单
                        snapshot_files = show_load_snapshot_menu()

            elif game_mode == 3:  # 残局加载菜单
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game_mode = 0  # 返回主菜单
                    elif event.key in [
                        pygame.K_1,
                        pygame.K_2,
                        pygame.K_3,
                        pygame.K_4,
                        pygame.K_5,
                        pygame.K_6,
                        pygame.K_7,
                        pygame.K_8,
                        pygame.K_9,
                        pygame.K_0,
                    ]:
                        # 选择残局文件
                        key_to_index = {
                            pygame.K_1: 0,
                            pygame.K_2: 1,
                            pygame.K_3: 2,
                            pygame.K_4: 3,
                            pygame.K_5: 4,
                            pygame.K_6: 5,
                            pygame.K_7: 6,
                            pygame.K_8: 7,
                            pygame.K_9: 8,
                            pygame.K_0: 9,
                        }
                        file_index = key_to_index[event.key]

                        if file_index < len(snapshot_files):
                            selected_file = snapshot_files[file_index]
                            loaded_game, loaded_mode = load_snapshot(selected_file)

                            if loaded_game is not None:
                                game = loaded_game
                                game_mode = loaded_mode
                                print(f"成功加载残局: {selected_file}")
                                print(
                                    f"游戏模式: {'双人对战' if game_mode == 1 else 'AI对战'}"
                                )
                                print(
                                    f"当前玩家: {'黑棋' if game.current_player == 1 else '白棋'}"
                                )
                                print(f"历史步数: {len(game.history)}")
                            else:
                                print("加载残局失败")
                                game_mode = 0
                        else:
                            print("无效的选择")
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
                            # 创建游戏副本用于保存
                            game_copy = copy.deepcopy(game)

                            # 如果是AI模式且当前轮到AI，需要悔棋AI的最后一步
                            if (
                                game_mode == 2
                                and game_copy.current_player == 1
                                and game_copy.history
                            ):
                                # 悔棋AI的最后一步
                                game_copy.undo_move()
                                print("已悔棋AI的最后一步，准备保存棋局")

                            snapshot = {
                                "current_player": game_copy.current_player,
                                "board": game_copy.board.tolist(),
                                "history": game_copy.history,
                                "game_mode": game_mode,
                                "game_over": game_copy.game_over,
                                "winner": game_copy.winner,
                                "save_time": datetime.datetime.now().isoformat(),
                                "total_moves": len(game_copy.history),
                                "ai_undone": game_mode == 2
                                and game_copy.current_player == 2,  # 标记是否悔棋了AI
                            }

                            # 确保board_snapshots目录存在
                            import os

                            os.makedirs("board_snapshots", exist_ok=True)

                            filename = f"board_snapshots/gomoku_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            with open(filename, "w", encoding="utf-8") as f:
                                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                            print(f"已保存棋局快照到 {filename}")
                            print(f"当前步数: {len(game_copy.history)}")
                            if snapshot["ai_undone"]:
                                print("注意：已悔棋AI的最后一步，保存的棋局轮到AI下棋")
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
        elif game_mode == 3:
            show_load_snapshot_menu()
        else:
            # AI线程启动与结果处理
            if game_mode == 2 and game.current_player == 2 and not game.game_over:
                if ai_thread is None or not ai_thread.is_alive():
                    if ai_move_result is None:
                        ai_think_start_time = datetime.datetime.now()
                        ai_think_time = None
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
                # 在上方一排显示信息
                current_player_text = (
                    f"当前: {'黑棋' if game.current_player == 1 else '白棋'}"
                )
                mode_text = f"模式: {'双人' if game_mode == 1 else 'AI'}"
                history_text = f"步数: {len(game.history)}"

                draw_text(current_player_text, (10, 10))
                draw_text(mode_text, (150, 10))
                draw_text(history_text, (250, 10))

                # AI思考中提示 - 放在按钮下方居中
                if (
                    game_mode == 2
                    and game.current_player == 2
                    and ai_thread is not None
                    and ai_thread.is_alive()
                ):
                    draw_text(
                        "AI思考中...",
                        (WINDOW_WIDTH // 2 - 50, button_y + button_height + 20),
                        (255, 0, 0),
                        font_type="middle",
                    )
                elif ai_think_time is not None:
                    # 显示AI思考用时
                    think_time_text = f"AI用时: {ai_think_time:.1f}秒"
                    draw_text(
                        think_time_text,
                        (WINDOW_WIDTH // 2 - 60, button_y + button_height + 20),
                        font_type="middle",
                    )
            pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
