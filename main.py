# main.py
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Set, Iterable

# ----------------------------
# Instance definition (generic)
# ----------------------------
@dataclass(frozen=True)
class Instance:
    name: str
    items: Tuple[str, ...]                 # e.g., ("W","G","C")
    forbidden_pairs: Tuple[Tuple[str,str], ...]  # e.g., (("W","G"),("G","C"))
    start_farmer: str                      # 'L' or 'R'
    start_item_side: Dict[str, str]        # map item -> 'L' or 'R'
    goal_side: str = 'R'                   # usually 'R'

# -------------
# WGC Instances
# -------------
W, G, C = "W", "G", "C"

INSTANCE_A = Instance(
    name="WGC A (classic)",
    items=(W, G, C),
    forbidden_pairs=((W, G), (G, C)),
    start_farmer='L',
    start_item_side={W: 'L', G: 'L', C: 'L'},
    goal_side='R',
)

INSTANCE_B = Instance(
    name="WGC B (goat starts right)",
    items=(W, G, C),
    forbidden_pairs=((W, G), (G, C)),
    start_farmer='L',
    start_item_side={W: 'L', G: 'R', C: 'L'},
    goal_side='R',
)

# -------------------
# Core state & helpers
# -------------------
State = Tuple[frozenset, str]  # (left_items, farmer_side 'L'|'R')
Action = Optional[str]         # human-readable move string or None

def make_start_state(inst: Instance) -> State:
    left_items = {i for i, side in inst.start_item_side.items() if side == 'L'}
    return (frozenset(left_items), inst.start_farmer)

def is_goal(state: State, inst: Instance) -> bool:
    left, _ = state
    # Goal: all items on goal_side (i.e., none on the opposite side)
    if inst.goal_side == 'R':
        return len(left) == 0
    else:
        # goal_side == 'L'
        return len(left) == len(inst.items)

def bank_unsafe(inst: Instance, bank: Set[str], farmer_here: bool) -> bool:
    """A bank is unsafe if farmer is NOT there and any forbidden pair is together."""
    if farmer_here:
        return False
    for a, b in inst.forbidden_pairs:
        if a in bank and b in bank:
            return True
    return False

def state_valid(state: State, inst: Instance) -> bool:
    left, farmer = state
    items_set = set(inst.items)
    right = items_set - set(left)
    farmer_left = (farmer == 'L')
    return (not bank_unsafe(inst, set(left), farmer_left)
            and not bank_unsafe(inst, right, not farmer_left))

def next_states(state: State, inst: Instance) -> List[Tuple[State, str]]:
    """
    Unit-cost moves: farmer crosses with zero or one item from his bank.
    """
    left, farmer = state
    items_set = set(inst.items)
    right = items_set - set(left)
    farmer_left = (farmer == 'L')
    current_bank = set(left) if farmer_left else set(right)

    candidates: List[Optional[str]] = [None] + sorted(list(current_bank))
    moves: List[Tuple[State, str]] = []

    for cargo in candidates:
        new_left = set(left)
        if farmer_left:  # L -> R
            if cargo is not None:
                new_left.remove(cargo)
            new_farmer = 'R'
            desc = f"Farmer takes {cargo or 'nothing'} L→R"
        else:            # R -> L
            if cargo is not None:
                new_left.add(cargo)
            new_farmer = 'L'
            desc = f"Farmer takes {cargo or 'nothing'} R→L"

        ns: State = (frozenset(new_left), new_farmer)
        if state_valid(ns, inst):
            moves.append((ns, desc))
    return moves

# ----------------
# Pretty formatting
# ----------------
def encode_state_lr(state: State, inst: Instance) -> str:
    """
    Encode as (F, item1, item2, ...) each 'L' or 'R' in fixed item order from inst.items.
    """
    left, farmer = state
    parts = [farmer]
    for it in inst.items:
        parts.append('L' if it in left else 'R')
    return "(" + ",".join(parts) + ")"

def label_action(desc: str) -> str:
    if "nothing" in desc:
        return "Return alone" if "R→L" in desc else "Go alone"
    if " W " in desc: return "Move Wolf"
    if " G " in desc: return "Move Goat"
    if " C " in desc: return "Move Cabbage"
    return desc

def print_clean_report(inst: Instance, algo_name: str,
                       path: List[Tuple[State, Action]], metrics: Dict[str, int]):
    print(f"Domain: WGC | Algorithm: {algo_name}")
    print(f"Solution cost: {metrics['solution_cost']} | Depth: {metrics['solution_depth']}")
    print(f"Nodes generated: {metrics['nodes_generated']} | Nodes expanded: {metrics['nodes_expanded']} | Max frontier: {metrics['max_frontier_size']}")
    print("Path:")
    for i in range(1, len(path)):
        prev_state = encode_state_lr(path[i-1][0], inst)
        next_state = encode_state_lr(path[i][0], inst)
        act = path[i][1] or ""
        label = label_action(act)
        print(f"  {i}) {label:<14} {prev_state} -> {next_state}")

# -----------
# BFS (graph)
# -----------
def bfs_solve_with_metrics(inst: Instance):
    start = make_start_state(inst)
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

        if is_goal(s, inst):
            path: List[Tuple[State, Action]] = []
            cur: Optional[State] = s
            while cur is not None:
                path.append((cur, action[cur]))
                cur = parent[cur]
            path.reverse()
            depth = len(path) - 1
            return path, {
                "nodes_generated": nodes_generated,
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier,
                "solution_depth": depth,
                "solution_cost": depth,
            }

        for ns, desc in next_states(s, inst):
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

# --------------------------
# IDS (DFS tree with cutoffs)
# --------------------------
def ids_solve_with_metrics(inst: Instance, max_depth_cap: int = 64):
    start = make_start_state(inst)

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

        if is_goal(node, inst):
            return node, False
        if depth == limit:
            return None, True

        cutoff_flag = False
        for ns, desc in next_states(node, inst):
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
            return path, {
                "nodes_generated": total_gen,
                "nodes_expanded": total_exp,
                "max_frontier_size": max_frontier,
                "solution_depth": depth,
                "solution_cost": depth,
                "depth_limit_used": limit,
            }
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

# -----------
# Menu / CLI
# -----------
def print_menu():
    print("\n=== Wolf–Goat–Cabbage Solver ===")
    print("1) BFS on Instance A")
    print("2) IDS on Instance A")
    print("3) BFS on Instance B")
    print("4) IDS on Instance B")
    print("5) Run BOTH (BFS A, IDS A, BFS B, IDS B)")
    print("6) Exit")

def run_and_report(inst: Instance, algo: str):
    if algo == "BFS":
        path, metrics = bfs_solve_with_metrics(inst)
    else:
        path, metrics = ids_solve_with_metrics(inst)

    print()
    if not path:
        print(f"Domain: WGC | Algorithm: {algo}\nNo solution found.")
    else:
        print_clean_report(inst, algo, path, metrics)

if __name__ == "__main__":
    while True:
        print_menu()
        choice = input("Choose an option (1-6): ").strip()
        if choice == "1":
            run_and_report(INSTANCE_A, "BFS")
        elif choice == "2":
            run_and_report(INSTANCE_A, "IDS")
        elif choice == "3":
            run_and_report(INSTANCE_B, "BFS")
        elif choice == "4":
            run_and_report(INSTANCE_B, "IDS")
        elif choice == "5":
            for inst, algo in [(INSTANCE_A, "BFS"), (INSTANCE_A, "IDS"),
                               (INSTANCE_B, "BFS"), (INSTANCE_B, "IDS")]:
                run_and_report(inst, algo)
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1–6.")
