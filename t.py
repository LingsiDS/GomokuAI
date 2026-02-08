"""
Gomoku AI (Python)
Alpha-Beta (Negamax) + Iterative Deepening + Transposition Table + Zobrist Hashing

Core logic only; plug your GUI or CLI around choose_move(...).
Tested with Python 3.9+.
"""

from __future__ import annotations
from dataclasses import dataclass
import time, random, math
from typing import List, Tuple, Optional, Dict

# =====================
# Parameters & constants
# =====================
from constants import N, WIN_LEN, INF, DEFAULT_MAX_DEPTH, DEFAULT_TIME_LIMIT_MS

EMPTY, BLACK, WHITE = 0, 1, 2


# =====================
# Zobrist hashing
# =====================
class Zobrist:
    def __init__(self, n: int):
        random.seed(20250821)
        self.table = [
            [[random.getrandbits(64) for _ in range(3)] for _ in range(n)]
            for _ in range(n)
        ]
        self.side_key = random.getrandbits(64)

    def piece_key(self, r: int, c: int, v: int) -> int:
        return self.table[r][c][v]


@dataclass
class TTEntry:
    key: int
    depth: int
    value: int
    flag: int  # 0=EXACT, 1=LOWER, 2=UPPER
    best_move: Optional[Tuple[int, int]]
    age: int


class TranspositionTable:
    EXACT, LOWER, UPPER = 0, 1, 2

    def __init__(self, size: int = 1 << 20):
        self.size = size
        self.table: Dict[int, TTEntry] = {}
        self.age = 0

    def new_search(self):
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
    ):
        # Replace by depth/age heuristic
        e = self.table.get(key)
        if (e is None) or (depth > e.depth) or (self.age > e.age + 2):
            self.table[key] = TTEntry(key, depth, value, flag, best_move, self.age)
        # Bound table size
        if len(self.table) > self.size * 1.1:
            # Simple aging cleanup
            drop_age = self.age - 2
            self.table = {k: v for k, v in self.table.items() if v.age >= drop_age}


# =====================
# Board representation
# =====================
class Board:
    def __init__(self, n: int = N):
        self.n = n
        self.grid = [[EMPTY] * n for _ in range(n)]
        self.turn = BLACK  # BLACK moves first by default
        self.zobrist = Zobrist(n)
        self.hash_key = 0
        self._init_hash()
        self.move_stack: List[Tuple[int, int, int]] = []  # (r,c,player)
        self.move_count = 0

    def _init_hash(self):
        self.hash_key = 0
        # empty board -> nothing to xor; side
        self.hash_key ^= self.zobrist.side_key if self.turn == BLACK else 0

    def at(self, r: int, c: int) -> int:
        return self.grid[r][c]

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def place(self, r: int, c: int):
        assert self.grid[r][c] == EMPTY
        v = self.turn
        self.grid[r][c] = v
        self.move_stack.append((r, c, v))
        self.move_count += 1
        # update hash
        self.hash_key ^= self.zobrist.piece_key(r, c, v)
        self.hash_key ^= self.zobrist.side_key  # toggle side
        # switch turn
        self.turn = BLACK + WHITE - self.turn

    def undo(self):
        r, c, v = self.move_stack.pop()
        self.turn = v  # because place() toggles
        self.hash_key ^= self.zobrist.side_key
        self.hash_key ^= self.zobrist.piece_key(r, c, v)
        self.grid[r][c] = EMPTY
        self.move_count -= 1

    def winner(self) -> int:
        # return BLACK/WHITE if someone has 5 in a row; else 0
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for r in range(self.n):
            for c in range(self.n):
                v = self.grid[r][c]
                if v == EMPTY:
                    continue
                for dr, dc in dirs:
                    cnt = 1
                    rr, cc = r + dr, c + dc
                    while self.inside(rr, cc) and self.grid[rr][cc] == v:
                        cnt += 1
                        if cnt >= WIN_LEN:
                            return v
                        rr += dr
                        cc += dc
        return EMPTY

    def full(self) -> bool:
        return self.move_count >= self.n * self.n

    def list_candidate_moves(self, radius: int = 2) -> List[Tuple[int, int]]:
        # Generate moves near existing stones to reduce branching
        if self.move_count == 0:
            center = self.n // 2
            return [(center, center)]
        mark = [[False] * self.n for _ in range(self.n)]
        has = False
        for r in range(self.n):
            for c in range(self.n):
                if self.grid[r][c] != EMPTY:
                    has = True
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            rr, cc = r + dr, c + dc
                            if self.inside(rr, cc) and self.grid[rr][cc] == EMPTY:
                                mark[rr][cc] = True
        if not has:
            center = self.n // 2
            return [(center, center)]
        moves = [(r, c) for r in range(self.n) for c in range(self.n) if mark[r][c]]
        return moves


# =====================
# Evaluation (simple, replace with your stronger pattern scoring)
# =====================
# We implement a lightweight pattern based heuristic:
#   - Immediate win/loss detection handled in search via winner()
#   - Count open/half-open 2/3/4-in-a-row for both sides in four directions
#   - Weights tuned coarsely

WEIGHTS = {
    "open4": 100000,
    "half4": 10000,
    "open3": 1200,
    "half3": 300,
    "open2": 60,
    "half2": 15,
}

from constants import DIRS


def evaluate(board: Board, me: int) -> int:
    opp = BLACK + WHITE - me
    win = board.winner()
    if win == me:
        return INF - 10
    if win == opp:
        return -INF + 10
    n = board.n

    def line_score(v: int) -> int:
        total = 0
        for dr, dc in DIRS:
            for r in range(n):
                for c in range(n):
                    # count segment starting here only if previous not same (avoid double)
                    pr, pc = r - dr, c - dc
                    if board.inside(pr, pc) and board.at(pr, pc) == v:
                        continue
                    # extend forward up to 5+ blanks to detect patterns roughly
                    seq = []
                    rr, cc = r, c
                    while board.inside(rr, cc):
                        seq.append(board.at(rr, cc))
                        rr += dr
                        cc += dc
                    # parse runs in seq
                    cnt = 0
                    i = 0
                    while i < len(seq):
                        if seq[i] == v:
                            j = i
                            while j < len(seq) and seq[j] == v:
                                j += 1
                            k = j - i
                            left_open = i - 1 >= 0 and seq[i - 1] == EMPTY
                            right_open = j < len(seq) and seq[j] == EMPTY
                            if k >= 4:
                                if left_open and right_open:
                                    total += WEIGHTS["open4"]
                                elif left_open or right_open:
                                    total += WEIGHTS["half4"]
                            elif k == 3:
                                if left_open and right_open:
                                    total += WEIGHTS["open3"]
                                elif left_open or right_open:
                                    total += WEIGHTS["half3"]
                            elif k == 2:
                                if left_open and right_open:
                                    total += WEIGHTS["open2"]
                                elif left_open or right_open:
                                    total += WEIGHTS["half2"]
                            i = j
                        else:
                            i += 1
        return total

    return line_score(me) - line_score(opp)


# =====================
# Move ordering helper
# =====================


def move_heuristic(board: Board, me: int, mv: Tuple[int, int]) -> int:
    r, c = mv
    # quick local pattern bump: favor center & adjacency
    center = board.n // 2
    dist = abs(r - center) + abs(c - center)
    adj = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if board.inside(rr, cc) and board.at(rr, cc) != EMPTY:
                adj += 1
    return -dist + 3 * adj


# =====================
# Search (Negamax + alpha-beta + TT + iterative deepening)
# =====================
class Searcher:
    def __init__(self):
        self.tt = TranspositionTable()
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit_ms = DEFAULT_TIME_LIMIT_MS
        self.stop = False
        self.best_move: Optional[Tuple[int, int]] = None
        self.last_score = 0

    def time_up(self) -> bool:
        return (time.monotonic() - self.start_time) * 1000.0 >= self.time_limit_ms

    def probe_tt(self, key: int, depth: int, alpha: int, beta: int) -> Optional[int]:
        e = self.tt.get(key)
        if e is None or e.depth < depth:
            return None
        if e.flag == TranspositionTable.EXACT:
            return e.value
        if e.flag == TranspositionTable.LOWER and e.value >= beta:
            return e.value
        if e.flag == TranspositionTable.UPPER and e.value <= alpha:
            return e.value
        return None

    def store_tt(
        self,
        key: int,
        depth: int,
        value: int,
        alpha: int,
        beta: int,
        best: Optional[Tuple[int, int]],
    ):
        flag = TranspositionTable.EXACT
        if value <= alpha:
            flag = TranspositionTable.UPPER
        elif value >= beta:
            flag = TranspositionTable.LOWER
        self.tt.put(key, depth, value, flag, best)

    def negamax(self, board: Board, depth: int, alpha: int, beta: int, me: int) -> int:
        if self.stop or self.time_up():
            self.stop = True
            return 0
        self.nodes += 1

        win = board.winner()
        if win == me:
            return INF - 10
        elif win == (BLACK + WHITE - me):
            return -INF + 10
        if depth == 0:
            return evaluate(board, me)

        # TT probe
        ttv = self.probe_tt(board.hash_key, depth, alpha, beta)
        if ttv is not None:
            return ttv

        # Generate & order moves
        moves = board.list_candidate_moves()
        # Try TT best move first for strong ordering
        tt_entry = self.tt.get(board.hash_key)
        tt_best = tt_entry.best_move if tt_entry else None
        if tt_best and tt_best in moves:
            moves.remove(tt_best)
            ordered = [tt_best] + moves
        else:
            ordered = moves
        ordered.sort(key=lambda mv: move_heuristic(board, me, mv), reverse=True)

        best_val = -INF
        best_move = None
        a0 = alpha
        opp = BLACK + WHITE - me

        for r, c in ordered:
            board.place(r, c)
            val = -self.negamax(board, depth - 1, -beta, -alpha, opp)
            board.undo()

            if self.stop:
                return 0
            if val > best_val:
                best_val = val
                best_move = (r, c)
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break

        self.store_tt(board.hash_key, depth, best_val, a0, beta, best_move)
        return best_val

    def iterative_deepening(
        self, board: Board, max_depth: int, time_limit_ms: int, me: int
    ) -> Tuple[Optional[Tuple[int, int]], int]:
        self.tt.new_search()
        self.nodes = 0
        self.start_time = time.monotonic()
        self.time_limit_ms = time_limit_ms
        self.stop = False
        self.best_move = None
        score = 0

        # Aspiration window parameters
        alpha = -INF
        beta = INF
        prev_score: Optional[int] = None

        for depth in range(1, max_depth + 1):
            if self.time_up():
                break
            # Narrow aspiration around previous score to speed cutoffs
            if prev_score is not None:
                window = 300  # tune
                alpha = prev_score - window
                beta = prev_score + window
            else:
                alpha, beta = -INF, INF

            s = self.negamax(board, depth, alpha, beta, me)

            # If fail-low/high, re-search with wide window (one re-search)
            if not self.stop and prev_score is not None and (s <= alpha or s >= beta):
                alpha, beta = -INF, INF
                s = self.negamax(board, depth, alpha, beta, me)

            if self.stop:
                break

            # Extract PV best move from TT
            e = self.tt.get(board.hash_key)
            if e and e.best_move:
                self.best_move = e.best_move
                score = s
                prev_score = s
            else:
                prev_score = s

        return self.best_move, score


# =====================
# Public API
# =====================


def choose_move(
    board: Board,
    time_limit_ms: int = DEFAULT_TIME_LIMIT_MS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Tuple[int, int]:
    """Return best move (r,c) for board.turn using ID+AB+TT under time limit."""
    me = board.turn
    searcher = Searcher()
    mv, score = searcher.iterative_deepening(board, max_depth, time_limit_ms, me)
    if mv is None:
        # Fallback: pick first candidate
        cand = board.list_candidate_moves()
        return cand[0] if cand else (-1, -1)
    return mv


# =====================
# Minimal CLI demo (optional)
# =====================
if __name__ == "__main__":
    b = Board(N)
    # Example: have AI make the first move
    r, c = choose_move(b, time_limit_ms=1000, max_depth=4)
    print("AI plays:", r, c)
