# wolf_goat_cabbage_search.py
from collections import deque
from typing import Dict, Optional, Tuple, List, Set

# --- Problem setup ---
W, G, C = "W", "G", "C"
ITEMS = {W, G, C}

State = Tuple[frozenset, str]  # (left_bank_items, 'L'|'R')
Action = Optional[str]         # e.g., "Farmer takes G L->R" or None for start

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
            desc = f"Farmer takes {cargo or 'nothing'} L->R"
        else:  # R -> L
            if cargo is not None:
                new_left.add(cargo)
            new_boat = 'L'
            desc = f"Farmer takes {cargo or 'nothing'} R->L"

        ns: State = (frozenset(new_left), new_boat)
        if state_valid(ns):
            moves.append((ns, desc))

    return moves

def pretty_state(state: State) -> str:
    left, boat = state
    right = ITEMS - set(left)
    return f"(Left={sorted(left)}, Right={sorted(right)}, Boat='{boat}')"

# --- BFS with metrics and repeated-state detection (global visited) ---
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

        left, _ = s
        if left == goal_left:
            # reconstruct
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

# --- IDS (Iterative Deepening Search) with metrics ---
def ids_solve_with_metrics(max_depth_cap: int = 64):
    """
    Iteratively increases depth limit: 0,1,2,... until solution found.
    Repeated-state detection is on the *current path only* (classic DFS tree search).
    Metrics are aggregated over all depth iterations.
    """
    start: State = (frozenset(ITEMS), 'L')
    goal_left = frozenset()

    total_nodes_generated = 0
    total_nodes_expanded  = 0
    max_frontier = 1  # measured as max recursion stack depth (call stack size)

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

    # Depth-Limited Search (recursive)
    def dls(node: State, depth: int, limit: int, path_set: Set[State]) -> Tuple[Optional[State], bool, int, int, int]:
        """
        Returns: (goal_state or None, cutoff, nodes_generated, nodes_expanded, max_stack_depth)
        cutoff=True means at least one path was cut by the depth limit.
        """
        nodes_expanded = 1  # counting this node expansion
        nodes_generated = 0
        local_max_stack = depth + 1  # stack size counting this frame

        # Goal test
        if node[0] == goal_left:
            return node, False, nodes_generated, nodes_expanded, local_max_stack

        if depth == limit:
            return None, True, nodes_generated, nodes_expanded, local_max_stack

        # Expand successors
        for ns, desc in next_states(node):
            nodes_generated += 1
            if ns in path_set:  # avoid cycles along current path
                continue
            parent[ns] = node
            action[ns] = desc
            path_set.add(ns)
            goal, cut, gen_c, exp_c, stack_c = dls(ns, depth + 1, limit, path_set)
            nodes_generated += gen_c
            nodes_expanded  += exp_c
            local_max_stack = max(local_max_stack, stack_c)
            path_set.remove(ns)
            if goal is not None:
                return goal, False, nodes_generated, nodes_expanded, local_max_stack
            # else continue exploring siblings; keep track of whether any child was cut off
            # We combine cutoff flags at the caller level
            cutoff_child = cut

        # If any child hit cutoff, signal cutoff upward
        # We can infer cutoff if depth < limit and no goal found but at least one branch hit the limit.
        # A simple heuristic: if limit > depth and we expanded any child but found no goal, treat as possible cutoff.
        return None, (depth < limit), nodes_generated, nodes_expanded, local_max_stack

    # Iterative deepening loop
    for limit in range(0, max_depth_cap + 1):
        parent.clear()
        action.clear()
        path_set = {start}
        goal, cutoff, gen_c, exp_c, stack_c = dls(start, 0, limit, path_set)

        total_nodes_generated += gen_c
        total_nodes_expanded  += exp_c
        max_frontier = max(max_frontier, stack_c)

        if goal is not None:
            path = reconstruct(goal)
            depth = len(path) - 1
            metrics = {
                "nodes_generated": total_nodes_generated,
                "nodes_expanded": total_nodes_expanded,
                "max_frontier_size": max_frontier,  # here: max recursion stack depth
                "solution_depth": depth,
                "solution_cost": depth,
                "depth_limit_used": limit,
            }
            return path, metrics

        if not cutoff:
            break  # no solution and no cutoff => exhausted tree under this cap

    return [], {
        "nodes_generated": total_nodes_generated,
        "nodes_expanded": total_nodes_expanded,
        "max_frontier_size": max_frontier,
        "solution_depth": -1,
        "solution_cost": -1,
        "depth_limit_used": None,
    }

# --- Run both solvers ---
if __name__ == "__main__":
    print("=== BFS ===")
    path_bfs, m_bfs = bfs_solve_with_metrics()
    if not path_bfs:
        print("No solution via BFS.\n")
    else:
        for i, (state, act) in enumerate(path_bfs):
            s = pretty_state(state)
            print(f"Step {i:>2}: (state={s}, action={act!r})")
        print("\n--- BFS Metrics ---")
        for k, v in m_bfs.items():
            print(f"{k.replace('_',' ').title()}: {v}")
        print()

    print("=== IDS ===")
    path_ids, m_ids = ids_solve_with_metrics()
    if not path_ids:
        print("No solution via IDS.\n")
    else:
        for i, (state, act) in enumerate(path_ids):
            s = pretty_state(state)
            print(f"Step {i:>2}: (state={s}, action={act!r})")
        print("\n--- IDS Metrics ---")
        for k, v in m_ids.items():
            print(f"{k.replace('_',' ').title()}: {v}")
