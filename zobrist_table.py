from dataclasses import dataclass
from constants import BOARD_SIZE
import random
from typing import Optional, Tuple, List, Dict


class Zobrist:
    def __init__(self) -> None:
        random.seed(20250821)
        self.table = [
            [[random.getrandbits(64) for _ in range(3)] for _ in range(BOARD_SIZE)]
            for _ in range(BOARD_SIZE)
        ]
        self.side_key = random.getrandbits(
            64
        )  # 表示当前局面该收下，有则表示先手轮次，否则表示当前为后手轮次

    def piece_key(self, row: int, col: int, value: int) -> int:
        return self.table[row][col][value]


@dataclass
class TTEntry:
    key: int
    depth: int  # 剩余的搜索深度
    value: int  # 该局面的评估值
    flag: int  # 0=EXACT, 1=LOWER, 2=UPPER，用于alpha-beta剪枝
    best_move: Optional[Tuple[int, int]]
    age: int  # 用于置换表的替换策略


class TranspositionTable:
    EXACT, LOWER, UPPER = 0, 1, 2

    def __init__(self, size: int = 1 << 20) -> None:
        self.size = size
        self.table: Dict[int, TTEntry] = {}
        self.age = 0

    def new_search(self) -> None:
        self.age += 1

    def get(self, key: int) -> Optional[TTEntry]:
        return self.table.get(key)

    def put(
        self,
        key: int,
        depth: int,
        value: int,
        flag: int,
        best_move: Optional[Tuple[int, int]],
    ) -> None:
        # Replace by depth/age heuristic
        # 如果该局面不存在，或者新搜索深度更大，或者旧条目太老，则覆盖
        e = self.table.get(key)
        if (e is None) or (depth > e.depth) or (self.age > e.age + 2):
            self.table[key] = TTEntry(key, depth, value, flag, best_move, self.age)
        # 控制表大小，超过容量时清理旧条目
        if len(self.table) > self.size * 1.1:
            drop_age = self.age - 2
            self.table = {k: v for k, v in self.table.items() if v.age >= drop_age}
