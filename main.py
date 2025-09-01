# main.py
from collections import deque
from typing import Dict, Optional, Tuple, List, Set

# --- Problem setup ---
W, G, C = "W", "G", "C"
ITEMS = {W, G, C}

State = Tuple[frozenset, str]  # (left_bank_items, 'L'|'R')
Action = Optional[str]         # e.g., "Farmer takes G L→R" or None for start

def bank_unsafe(bank: Set[str], farmer_here: bool) -> bool:
    if farmer_here:
        return False
    return ({W, G} <= bank) or ({G, C} <= bank)

def state_valid(state: State) -> bool:
    left, boat = state
    right = ITEMS - set(left)
    farmer_left = (boat == 'L')
    return (not bank_unsafe(set(left), farmer_left)
            and not bank_unsafe(right, not farmer_left))

def next_states(state: State) -> List[Tuple[State, str]]:
    left, boat = state
    right = ITEMS - set(left)
    farmer_left = (boat == 'L')
    current_bank = set(left) if farmer_left else set(right)

    candidates = [None] + sorted(list(current_bank))  # None = take nothing
    moves: List[Tuple[State, str]] = []

    for cargo in candidates:
        new_left = set(left)
        if farmer_left:  # L -> R
            if cargo is not None:
                new_left.remove(cargo)
            new_boat = 'R'
            desc = f"Farmer takes {cargo or 'nothing'} L→R"
        else:  # R -> L
            if cargo is not None:
                new_left.add(cargo)
            new_boat = 'L'
            desc = f"Farmer takes {cargo or 'nothing'} R→L"

        ns: State = (frozenset(new_left), new_boat)
        if state_valid(ns):
            moves.append((ns, desc))
    return moves

# --- Solvers ---
def bfs_solve_with_metrics():
    start: State = (frozenset(ITEMS), 'L')
    goal_left = frozenset()

    q: deque[State] = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    action: Dict[State, Action] = {start: None}
    visited: Set[State] = {start}

    nodes_generated = 0
    nodes_expanded  = 0
    max_frontier    = 1

    while q:
        s = q.popleft()
        nodes_expanded += 1

        if s[0] == goal_left:
            # reconstruct path
            path: List[Tuple[State, Action]] = []
            cur: Optional[State] = s
            while cur is not None:
                path.append((cur, action[cur]))
                cur = parent[cur]
            path.reverse()
            depth = len(path) - 1
            metrics = {
                "nodes_generated": nodes_generated,
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier,
                "solution_depth": depth,
                "solution_cost": depth,
            }
            return path, metrics

        for ns, desc in next_states(s):
            nodes_generated += 1
            if ns not in visited:
                visited.add(ns)
                parent[ns] = s
                action[ns] = desc
                q.append(ns)
                if len(q) > max_frontier:
                    max_frontier = len(q)

    return [], {
        "nodes_generated": nodes_generated,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier,
        "solution_depth": -1,
        "solution_cost": -1,
    }

def ids_solve_with_metrics(max_depth_cap: int = 64):
    """
    IDS with path-based repeated-state avoidance (DFS tree search).
    Frontier size is the max recursion stack depth.
    """
    start: State = (frozenset(ITEMS), 'L')
    goal_left = frozenset()

    total_gen = 0
    total_exp = 0
    max_frontier = 1

    parent: Dict[State, Optional[State]] = {}
    action: Dict[State, Action] = {}

    def reconstruct(goal: State) -> List[Tuple[State, Action]]:
        path: List[Tuple[State, Action]] = []
        cur: Optional[State] = goal
        while cur is not None:
            path.append((cur, action.get(cur)))
            cur = parent.get(cur)
        path.reverse()
        return path

    def dls(node: State, depth: int, limit: int, path_set: Set[State]):
        nonlocal total_gen, total_exp, max_frontier
        total_exp += 1
        max_frontier = max(max_frontier, depth + 1)

        if node[0] == goal_left:
            return node, False
        if depth == limit:
            return None, True

        cutoff_flag = False
        for ns, desc in next_states(node):
            total_gen += 1
            if ns in path_set:
                continue
            parent[ns] = node
            action[ns] = desc
            path_set.add(ns)
            goal, cut = dls(ns, depth + 1, limit, path_set)
            path_set.remove(ns)
            if goal:
                return goal, False
            if cut:
                cutoff_flag = True
        return None, cutoff_flag

    for limit in range(max_depth_cap + 1):
        parent.clear()
        action.clear()
        goal, cutoff = dls(start, 0, limit, {start})
        if goal:
            path = reconstruct(goal)
            depth = len(path) - 1
            metrics = {
                "nodes_generated": total_gen,
                "nodes_expanded": total_exp,
                "max_frontier_size": max_frontier,
                "solution_depth": depth,
                "solution_cost": depth,
                "depth_limit_used": limit,
            }
            return path, metrics
        if not cutoff:
            break

    return [], {
        "nodes_generated": total_gen,
        "nodes_expanded": total_exp,
        "max_frontier_size": max_frontier,
        "solution_depth": -1,
        "solution_cost": -1,
        "depth_limit_used": None,
    }

# --- Formatting helpers (clean output) ---
def encode_state_lr(state: State) -> str:
    """
    Encode state as (F,W,G,C) each 'L' or 'R' for Left/Right banks.
    F = farmer (boat side), W/G/C = item side
    """
    left, boat = state
    farmer = boat
    wolf   = 'L' if W in left else 'R'
    goat   = 'L' if G in left else 'R'
    cab    = 'L' if C in left else 'R'
    return f"({farmer},{wolf},{goat},{cab})"

def label_action(desc: str) -> str:
    """
    Map verbose descriptions to short labels:
      "Farmer takes G L→R" -> "Move Goat"
      "Farmer takes W R→L" -> "Move Wolf"
      "Farmer takes C L→R" -> "Move Cabbage"
      "Farmer takes nothing L→R" -> "Go alone"
      "Farmer takes nothing R→L" -> "Return alone"
    """
    if "nothing" in desc:
        return "Return alone" if "R→L" in desc else "Go alone"
    if " G " in desc:
        return "Move Goat"
    if " W " in desc:
        return "Move Wolf"
    if " C " in desc:
        return "Move Cabbage"
    return desc  # fallback

def print_clean_report(algorithm_name: str, path: List[Tuple[State, Action]], metrics: Dict[str, int]):
    domain = "WGC"
    print(f"Domain: {domain} | Algorithm: {algorithm_name}")
    print(f"Solution cost: {metrics['solution_cost']} | Depth: {metrics['solution_depth']}")
    print(f"Nodes generated: {metrics['nodes_generated']} | Nodes expanded: {metrics['nodes_expanded']} | Max frontier: {metrics['max_frontier_size']}")
    print("Path:")
    # Print transitions (state_i -> state_{i+1})
    for i in range(1, len(path)):
        prev_state = encode_state_lr(path[i-1][0])
        next_state = encode_state_lr(path[i][0])
        act = path[i][1] or ""
        label = label_action(act)
        print(f"  {i}) {label:<14} {prev_state} -> {next_state}")

# --- Simple Menu UI ---
def print_menu():
    print("\n=== Wolf–Goat–Cabbage Solver ===")
    print("1) Run BFS")
    print("2) Run IDS")
    print("3) Exit")

if __name__ == "__main__":
    while True:
        print_menu()
        choice = input("Choose an option (1-3): ").strip()
        if choice == "1":
            path, metrics = bfs_solve_with_metrics()
            print()
            if not path:
                print("Domain: WGC | Algorithm: BFS\nNo solution found.")
            else:
                print_clean_report("BFS", path, metrics)
        elif choice == "2":
            path, metrics = ids_solve_with_metrics()
            print()
            if not path:
                print("Domain: WGC | Algorithm: IDS\nNo solution found.")
            else:
                print_clean_report("IDS", path, metrics)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
