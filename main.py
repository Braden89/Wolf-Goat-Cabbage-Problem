W, G, C = "W", "G", "C"
ITEMS = {W, G, C}

def back_unsafe(bank: set, farmer_here:bool) -> bool:
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

if __name__ == "__main__":
    # Some quick checks:
    start = (frozenset({W, G, C}), 'L')  # all on left, farmer left
    print("Start valid?", state_valid(start))  # should be True

    bad = (frozenset({W, G}), 'R')  # W+G alone on left (farmer right) -> invalid
    print("W+G alone on left valid?", state_valid(bad))  # should be False

    ok = (frozenset({G}), 'R')  # W+C together on right with farmer -> ok
    print("G left, farmer right valid?", state_valid(ok))  # should be True
