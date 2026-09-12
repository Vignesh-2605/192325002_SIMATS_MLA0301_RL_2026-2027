from collections import deque

import numpy as np

N = 4                      # board is N x N
NSQ = N * N
CAPTURED = -1              # sentinel for a rook that has been taken

WHITE, BLACK = 0, 1

GAMMA = 0.99
STEP_COST = -0.01
WIN_REWARD = 1.0
DRAW_REWARD = -1.0
THETA = 1e-9               # value-iteration convergence threshold


def rc(sq):
    """Square index -> (row, col)."""
    return divmod(sq, N)


def sq(r, c):
    """(row, col) -> square index."""
    return r * N + c


def on_board(r, c):
    return 0 <= r < N and 0 <= c < N


KING_DELTAS = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]

ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def king_moves(from_sq):
    r, c = rc(from_sq)
    out = []
    for dr, dc in KING_DELTAS:
        nr, nc = r + dr, c + dc
        if on_board(nr, nc):
            out.append(sq(nr, nc))
    return out


# precompute king neighbourhoods once; used constantly in legality checks
KING_NEIGHBOURS = [set(king_moves(s)) for s in range(NSQ)]


def kings_adjacent(a, b):
    return b in KING_NEIGHBOURS[a]


def rook_attacks(rook, blockers):
    """Squares attacked by a rook, stopping at (and including) the first blocker."""
    if rook == CAPTURED:
        return set()
    r, c = rc(rook)
    out = set()
    for dr, dc in ROOK_DIRS:
        nr, nc = r + dr, c + dc
        while on_board(nr, nc):
            s = sq(nr, nc)
            out.add(s)
            if s in blockers:
                break
            nr, nc = nr + dr, nc + dc
    return out


def white_attacks(wk, wr, bk):
    """Every square White controls. The black king does not block the rook's
    line of attack through it, but for our purposes treating it as a blocker
    is correct because a square behind the king is still 'attacked through'
    only for check-evasion, which 4x4 endgames do not need."""
    blockers = {wk}
    if bk is not None:
        blockers.add(bk)
    return KING_NEIGHBOURS[wk] | rook_attacks(wr, blockers)


# --------------------------------------------------------------------------
# state legality and move generation
# --------------------------------------------------------------------------
def legal_position(wk, wr, bk):
    """A position is legal if no two pieces share a square and the kings are
    not adjacent."""
    if wk == bk:
        return False
    if wr != CAPTURED and (wr == wk or wr == bk):
        return False
    if kings_adjacent(wk, bk):
        return False
    return True


def black_in_check(wk, wr, bk):
    return bk in white_attacks(wk, wr, bk)


def white_moves(wk, wr, bk):
    """All legal White moves from (wk, wr, bk) with White to move.

    Returns a list of (label, new_wk, new_wr).
    """
    moves = []

    # king moves: may not step next to the black king, nor onto an occupied square
    for dst in king_moves(wk):
        if dst == bk:
            continue                      # capturing the bare king is not a move
        if wr != CAPTURED and dst == wr:
            continue
        if kings_adjacent(dst, bk):
            continue
        moves.append((f"K{wk}->{dst}", dst, wr))

    # rook moves: slide until blocked; cannot pass through or land on either king
    if wr != CAPTURED:
        r, c = rc(wr)
        for dr, dc in ROOK_DIRS:
            nr, nc = r + dr, c + dc
            while on_board(nr, nc):
                dst = sq(nr, nc)
                if dst == wk or dst == bk:
                    break                 # own king blocks; enemy king cannot be taken
                moves.append((f"R{wr}->{dst}", wk, dst))
                nr, nc = nr + dr, nc + dc

    return moves


def black_moves(wk, wr, bk):
    """All legal Black king moves. Black may capture the rook only if the rook
    is not defended by the white king."""
    moves = []
    for dst in king_moves(bk):
        if dst == wk:
            continue
        if kings_adjacent(dst, wk):
            continue                      # cannot move adjacent to the enemy king
        if wr != CAPTURED and dst == wr:
            # capture is legal only when the rook is undefended
            if wr in KING_NEIGHBOURS[wk]:
                continue
            moves.append((f"k{bk}x{dst}", CAPTURED, dst))
            continue
        # cannot move into a square the rook attacks
        if wr != CAPTURED and dst in rook_attacks(wr, {wk, bk}):
            continue
        moves.append((f"k{bk}->{dst}", wr, dst))
    return moves


# --------------------------------------------------------------------------
# terminal classification
# --------------------------------------------------------------------------
def terminal_reward(state):
    """Return (is_terminal, reward) for a state with Black to move."""
    wk, wr, bk, turn = state
    if wr == CAPTURED:
        return True, DRAW_REWARD          # bare kings: cannot force mate
    if turn != BLACK:
        return False, 0.0
    if black_moves(wk, wr, bk):
        return False, 0.0
    return (True, WIN_REWARD) if black_in_check(wk, wr, bk) else (True, DRAW_REWARD)


# --------------------------------------------------------------------------
# state space enumeration
# --------------------------------------------------------------------------
def enumerate_states():
    """Breadth-first sweep over every reachable legal state, White to move first."""
    states = set()
    frontier = deque()

    for wk in range(NSQ):
        for wr in range(NSQ):
            for bk in range(NSQ):
                if legal_position(wk, wr, bk) and not black_in_check(wk, wr, bk):
                    s = (wk, wr, bk, WHITE)
                    if s not in states:
                        states.add(s)
                        frontier.append(s)

    while frontier:
        wk, wr, bk, turn = frontier.popleft()
        if turn == WHITE:
            successors = [(nwk, nwr, bk, BLACK) for _, nwk, nwr in white_moves(wk, wr, bk)]
        else:
            done, _ = terminal_reward((wk, wr, bk, turn))
            successors = [] if done else [(wk, nwr, nbk, WHITE)
                                          for _, nwr, nbk in black_moves(wk, wr, bk)]
        for s in successors:
            if s not in states:
                states.add(s)
                frontier.append(s)

    return sorted(states)


# --------------------------------------------------------------------------
# transition model:  White action -> distribution over next White-to-move states
# --------------------------------------------------------------------------
def transitions(state):
    """For a White-to-move state, return {action_label: (reward, [(prob, s'), ...])}.

    Black's uniformly random reply is folded into the transition, so the agent
    faces a single-player MDP whose dynamics already contain the opponent.
    """
    wk, wr, bk, _ = state
    model = {}

    for label, nwk, nwr in white_moves(wk, wr, bk):
        after_white = (nwk, nwr, bk, BLACK)
        done, term_r = terminal_reward(after_white)
        if done:
            model[label] = (STEP_COST + term_r, [])
            continue

        replies = black_moves(nwk, nwr, bk)
        p = 1.0 / len(replies)
        outcomes = []
        for _, rwr, rbk in replies:
            nxt = (nwk, rwr, rbk, WHITE)
            outcomes.append((p, nxt))
        model[label] = (STEP_COST, outcomes)

    return model


def build_model(states):
    white_states = [s for s in states if s[3] == WHITE]
    model = {s: transitions(s) for s in white_states}
    return white_states, model


# --------------------------------------------------------------------------
# value iteration
# --------------------------------------------------------------------------
def value_iteration(white_states, model, max_sweeps=2000):
    V = {s: 0.0 for s in white_states}

    for sweep in range(max_sweeps):
        delta = 0.0
        for s in white_states:
            actions = model[s]
            if not actions:
                new_v = DRAW_REWARD          # White has no legal move
            else:
                best = -np.inf
                for _, (r, outcomes) in actions.items():
                    q = r + GAMMA * sum(p * V.get(nxt, DRAW_REWARD) for p, nxt in outcomes)
                    best = max(best, q)
                new_v = best
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < THETA:
            print(f"Value iteration converged after {sweep + 1} sweeps "
                  f"(delta = {delta:.3e})")
            break
    else:
        print(f"Stopped at the sweep limit with delta = {delta:.3e}")

    return V


def greedy_policy(white_states, model, V):
    policy = {}
    for s in white_states:
        actions = model[s]
        if not actions:
            continue
        best_a, best_q = None, -np.inf
        for a, (r, outcomes) in actions.items():
            q = r + GAMMA * sum(p * V.get(nxt, DRAW_REWARD) for p, nxt in outcomes)
            if q > best_q:
                best_a, best_q = a, q
        policy[s] = best_a
    return policy


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def render(state):
    wk, wr, bk, turn = state
    grid = [["." for _ in range(N)] for _ in range(N)]
    r, c = rc(wk); grid[r][c] = "K"
    if wr != CAPTURED:
        r, c = rc(wr); grid[r][c] = "R"
    r, c = rc(bk); grid[r][c] = "k"
    side = "White" if turn == WHITE else "Black"
    return "\n".join(" ".join(row) for row in grid) + f"\n{side} to move"


def play_optimal_line(start, model, policy, rng, max_plies=40):
    """Follow the greedy policy against a uniformly random Black and print the line."""
    state = start
    print("\nSample optimal line (Black replies uniformly at random)")
    print(render(state))

    for ply in range(max_plies):
        action = policy.get(state)
        if action is None:
            print("\nWhite has no legal move.")
            return
        wk, wr, bk, _ = state
        move = next(m for m in white_moves(wk, wr, bk) if m[0] == action)
        _, nwk, nwr = move
        after_white = (nwk, nwr, bk, BLACK)
        print(f"\nply {ply + 1}: White plays {action}")

        done, reward = terminal_reward(after_white)
        if done:
            outcome = "CHECKMATE" if reward == WIN_REWARD else "DRAW"
            print(render(after_white))
            print(f"\nResult: {outcome} after {ply + 1} White moves.")
            return

        replies = black_moves(nwk, nwr, bk)
        label, rwr, rbk = replies[rng.integers(len(replies))]
        print(f"         Black replies {label}")
        state = (nwk, rwr, rbk, WHITE)
        print(render(state))

        if state[1] == CAPTURED:
            print("\nResult: rook captured, mate can no longer be forced.")
            return

    print("\nMove limit reached.")


def main():
    rng = np.random.default_rng(0)

    print("Simplified chess MDP — 4x4 board, K+R vs k")
    print("=" * 55)

    states = enumerate_states()
    white_states, model = build_model(states)
    print(f"Reachable states            : {len(states)}")
    print(f"White-to-move decision states: {len(white_states)}")

    V = value_iteration(white_states, model)
    policy = greedy_policy(white_states, model, V)

    values = np.array(list(V.values()))
    print(f"\nState-value summary  min={values.min():+.3f}  "
          f"mean={values.mean():+.3f}  max={values.max():+.3f}")

    winning = [s for s in white_states if V[s] > 0.5]
    print(f"States from which mate is forced: {len(winning)} "
          f"({100 * len(winning) / len(white_states):.1f}% of decision states)")

    best = max(white_states, key=lambda s: V[s])
    print(f"\nHighest-valued state V = {V[best]:+.4f}")
    print(render(best))

    play_optimal_line(best, model, policy, rng)


if __name__ == "__main__":
    main()
