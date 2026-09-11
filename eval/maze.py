"""Task2 走迷宫: 固定seed生成迷宫, BFS最短路为参考, 判定=路径合法且到达终点."""
import random
from collections import deque

WALL, OPEN, START, END = "#", ".", "S", "E"
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

def gen_maze(n, seed):
    """随机DFS生成完美迷宫(n为奇数), 再随机打通约15%墙体形成多路径."""
    rng = random.Random(seed)
    g = [[WALL] * n for _ in range(n)]
    stack = [(1, 1)]; g[1][1] = OPEN
    while stack:
        r, c = stack[-1]
        nbs = []
        for dr, dc in MOVES.values():
            nr, nc = r + 2 * dr, c + 2 * dc
            if 1 <= nr < n - 1 and 1 <= nc < n - 1 and g[nr][nc] == WALL:
                nbs.append((dr, dc))
        if not nbs:
            stack.pop(); continue
        dr, dc = rng.choice(nbs)
        g[r + dr][c + dc] = OPEN
        nr, nc = r + 2 * dr, c + 2 * dc
        g[nr][nc] = OPEN
        stack.append((nr, nc))
    for _ in range(int(n * n * 0.15)):
        r, c = rng.randrange(1, n - 1), rng.randrange(1, n - 1)
        if g[r][c] == WALL:
            g[r][c] = OPEN
    g[1][1] = START; g[n - 2][n - 2] = END
    return g

def to_text(g):
    return "\n".join("".join(row) for row in g)

def find(g, ch):
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == ch: return (r, c)
    raise ValueError(ch)

def bfs_path(g):
    n = len(g); s, e = find(g, START), find(g, END)
    prev = {s: None}; dq = deque([s])
    while dq:
        cur = dq.popleft()
        if cur == e: break
        for m, (dr, dc) in MOVES.items():
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < n and 0 <= nb[1] < n and g[nb[0]][nb[1]] != WALL and nb not in prev:
                prev[nb] = (cur, m); dq.append(nb)
    if e not in prev: return None
    path = []; cur = e
    while prev[cur] is not None:
        cur, m = prev[cur]; path.append(m)
    return "".join(reversed(path))

def check_path(g, moves):
    """返回 (ok, 走到哪, 步数): 全程无撞墙且终点为E才算通过."""
    n = len(g); cur = find(g, START); steps = 0
    for ch in moves.upper():
        if ch not in MOVES: continue
        dr, dc = MOVES[ch]
        nb = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nb[0] < n and 0 <= nb[1] < n) or g[nb[0]][nb[1]] == WALL:
            return False, cur, steps
        cur = nb; steps += 1
    r, c = cur
    return (g[r][c] == END), cur, steps

def extract_moves(text):
    """取文本最后一个只含UDLR的token为路径."""
    import re
    cands = re.findall(r"\b[UDLRudlr]{2,}\b", text)
    return cands[-1].upper() if cands else ""

def prompt_for(g):
    n = len(g)
    return (f"Maze {n}x{n} (S=start, E=end, #=wall, .=open path):\n{to_text(g)}\n"
            f"Move with U(up) D(down) L(left) R(right). End your answer with the move sequence only.\nAnswer:")

def load_mazes():
    """20道: 5x5 x10 + 7x7 x10, seed固定可复现."""
    items = []
    for i in range(10):
        g = gen_maze(5, seed=1000 + i)
        items.append({"id": f"maze5-{i}", "size": 5, "grid": to_text(g), "ref_len": len(bfs_path(g))})
    for i in range(10):
        g = gen_maze(7, seed=2000 + i)
        items.append({"id": f"maze7-{i}", "size": 7, "grid": to_text(g), "ref_len": len(bfs_path(g))})
    return items

if __name__ == "__main__":
    # 自检: 生成合法性 + BFS可达 + 判定正反例
    for n, seed in [(5, 1000), (7, 2000), (5, 1007)]:
        g = gen_maze(n, seed)
        ref = bfs_path(g)
        assert ref, f"{n}x{n} seed={seed} 不可达"
        ok, _, steps = check_path(g, ref)
        assert ok, "参考路径判定失败"
        bad, _, _ = check_path(g, "UUUU")
        assert not bad, "撞墙路径误判通过"
        print(f"{n}x{n} seed={seed}: 参考最短 {len(ref)} 步, 判定OK")
    ms = load_mazes()
    print(f"题集 {len(ms)} 道, 参考步数范围: {min(m['ref_len'] for m in ms)}-{max(m['ref_len'] for m in ms)}")
    print("MAZE SELFTEST PASS")
