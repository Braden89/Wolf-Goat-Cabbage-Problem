W, G, C = "W", "G", "C"
ITEMS = {W, G, C}

def bank_unsafe(bank: set, farmer_here:bool) -> bool:
    """
    A bank is unsafe if the farmer is NOT there and (W+G) or (G+C) are together
    return true if their is danger and false if their is none.
    """
    if farmer_here:
        return False
    return ({W, G} <= bank) or ({G, C} <= bank)

def state_valid(state) -> bool:
    """
    state = (frozenset(left_bank_items), boat_side)
    Return True iff neither bank has a forbidden pair without the farmer.
    """
    left, boat = state
    right = ITEMS - set(left)
    farmer_left = (boat == 'L')
    return (not bank_unsafe(set(left), farmer_left)
            and not bank_unsafe(right, not farmer_left))

def next_states(state) -> bool:
    """
    Return list of (new_state, move_description).
    state = (frozenset(left_bank_items), 'L' or 'R')
    """
    left, boat = state
    right = ITEMS - set(left)
    farmer_left = (boat == 'L')
    current_bank = set(left) if farmer_left else set(right)

    candidates = [None] + sorted(list(current_bank)) # take northing or one item
    moves = []

    for cargo in candidates:
        new_left = set(left)
        if farmer_left: #L -> R
            if cargo is not None:
                new_left.remove(cargo)
            new_boat = 'R'
            desc = f"Farmer takes {cargo or 'nothing'} L->R"
        else: # L -> R
            if cargo is not None:
                new_left.add(cargo)
            new_boat = 'L'
            desc = f"Farmers takes {cargo or 'nothing'} R->L"

        ns = (frozenset(new_left), new_boat)
        if state_valid(ns):
            moves.append((ns, desc))

    return moves

if __name__ == "__main__":
    start = (frozenset({W, G, C}), 'L')
    print("From the start, legal moves are:")
    for ns, d in next_states(start):
        print(" -", d, "→", ns)