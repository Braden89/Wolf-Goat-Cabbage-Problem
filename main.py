# wolf_goat_cabbage_bfs_metrics.py
from collections import deque
from typing import Dict, Optional, Tuple, List, Set

# --- Problem setup ---
W, G, C = "W", "G", "C"
ITEMS = {W, G, C}

State = Tuple[frozenset, str]  # (left_bank_items, 'L'|'R')
Action = Optional[str]         # e.g., "Farmer takes G L→R" or None for start

def bank_unsafe(bank: Set[str], farmer_here: bool) -> bool:
    """A bank is unsafe if the farmer is NOT there and (W+G) or (G+C) are together."""
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
    """
    Generate all legal next states with action descriptions.
    Farmer always moves (boat moves), taking either nothing or exactly one item.
    """
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

def bfs_solve_with_metrics() -> Tuple[List[Tuple[State, Action]], Dict[str, int]]:
    """
    Standard BFS with repeated-state detection and useful metrics.
    Returns:
      - path: list of (state, action) from start to goal (action is None for start)
      - metrics: dict with counts
    """
    start: State = (frozenset(ITEMS), 'L')
    goal_left = frozenset()

    q: deque[State] = deque([start])
    parent: Dict[State, Optional[State]] = {start: None}
    action: Dict[State, Action] = {start: None}
    visited: Set[State] = {start}

    # Metrics
    nodes_generated = 0      # legal successors considered (before duplicate check)
    nodes_expanded  = 0      # states actually popped and expanded
    max_frontier    = 1      # track peak queue length

    while q:
        s = q.popleft()
        nodes_expanded += 1

        left, _ = s
        if left == goal_left:
            # reconstruct path
            path: List[Tuple[State, Action]] = []
            cur: Optional[State] = s
            while cur is not None:
                path.append((cur, action[cur]))
                cur = parent[cur]
            path.reverse()

            depth = len(path) - 1
            cost = depth  # unit cost per action
            metrics = {
                "nodes_generated": nodes_generated,
                "nodes_expanded": nodes_expanded,
                "max_frontier_size": max_frontier,
                "solution_depth": depth,
                "solution_cost": cost,
            }
            return path, metrics

        # Expand successors
        for ns, desc in next_states(s):
            nodes_generated += 1  # count every legal successor we consider
            if ns not in visited:
                visited.add(ns)
                parent[ns] = s
                action[ns] = desc
                q.append(ns)
                if len(q) > max_frontier:
                    max_frontier = len(q)

    # No solution (shouldn't happen for this puzzle)
    return [], {
        "nodes_generated": nodes_generated,
        "nodes_expanded": nodes_expanded,
        "max_frontier_size": max_frontier,
        "solution_depth": -1,
        "solution_cost": -1,
    }

def pretty_state(state: State) -> str:
    left, boat = state
    right = ITEMS - set(left)
    return f"(Left={sorted(left)}, Right={sorted(right)}, Boat='{boat}')"

if __name__ == "__main__":
    path, metrics = bfs_solve_with_metrics()
    if not path:
        print("No solution found.")
    else:
        print("Solution path as (state, action) steps:\n")
        for i, (state, act) in enumerate(path):
            s = pretty_state(state)
            if act is None:
                print(f"Step {i:>2}: (state={s}, action=None)   # Start")
            else:
                print(f"Step {i:>2}: (state={s}, action='{act}')")

        print("\n--- Search Metrics ---")
        for k, v in metrics.items():
            print(f"{k.replace('_', ' ').title()}: {v}")
