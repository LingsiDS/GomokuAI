# ==================== 五子棋项目常量配置文件 ====================
# 此文件包含项目中所有用到的常量值，方便在多个文件之间共享

# ==================== 游戏核心常量 ====================
# 棋盘大小
BOARD_SIZE = 15
N = 15  # 与BOARD_SIZE相同，保持兼容性

# 胜利条件
WIN_LEN = 5  # 连成5子获胜

# ==================== 窗口和UI常量 ====================
# 窗口设置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 900  # 增加窗口高度，为按钮留出空间

# 棋盘绘制相关
CELL_SIZE = 50
MARGIN = (WINDOW_WIDTH - (BOARD_SIZE - 1) * CELL_SIZE) // 2
BOARD_BOTTOM = MARGIN + (BOARD_SIZE - 1) * CELL_SIZE  # 棋盘底部位置

# ==================== 颜色定义 ====================
# 基础颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# 棋盘和UI颜色
BOARD_COLOR = (194, 178, 128)
HIGHLIGHT_COLOR = (255, 0, 0, 100)
TIP_COLOR_BLACK = (0, 0, 0, 100)
TIP_COLOR_WHITE = (255, 255, 255, 100)

# 按钮颜色
BUTTON_COLOR = (100, 150, 200)
BUTTON_HOVER_COLOR = (120, 170, 220)
BUTTON_TEXT_COLOR = WHITE

# 棋子颜色
BLACK_STONE = BLACK
WHITE_STONE = WHITE

# ==================== AI算法常量 ====================
# 搜索相关
INF = 10**9  # 无穷大值
DEFAULT_MAX_DEPTH = 6  # 默认最大搜索深度
DEFAULT_TIME_LIMIT_MS = 1500  # 默认时间限制（毫秒）

# 方向向量（用于模式匹配）
DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# ==================== 评分权重常量 ====================
# 这些权重用于AI评估函数
WEIGHTS = {
    # 可以根据需要添加更多权重配置
}

# ==================== 正则表达式模式常量 ====================
# 这些常量在game_eval.py中定义，但为了完整性在这里列出
# 实际使用时需要从game_eval模块导入
# BLACK_UNION = re.compile(...)
# WHITE_UNION = re.compile(...)

# ==================== 游戏状态常量 ====================
# 玩家标识
PLAYER_BLACK = 1
PLAYER_WHITE = 2

# 游戏状态
GAME_ACTIVE = 0
GAME_OVER = 1
