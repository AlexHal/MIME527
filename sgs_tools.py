"""
sgs_tools.py - supporting code for the MIME 525 notebooks.

You do not need to read this file to do the exercises, but you are welcome to.
It holds the functions that the notebooks hand to several CPU cores at once.
(Python can only send a function to another core if it lives in a .py file like this one.)

What is inside
    count(n)                      the counting loop from Exercises 1-4
    sgs_1d(seed)                  a tiny 1-D Sequential Gaussian Simulation (one drillhole line)
    sgs_2d(seed)                  a small 2-D SGS of Cu grade on a 40 x 40 block model (1 km x 1 km, 25 m blocks)
    ore_summary(seed)             simulate one 2-D model and return its ore tonnes and metal
    mine_npv(grade, cutoff, stockpile_cutoff, seed)
                                  a toy mine -> stockpile -> plant simulator, returns NPV in $M

These are teaching models: small, fast and easy to read, not real mine-planning tools.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Exercises 1-4: the counting loop
# ---------------------------------------------------------------------------
def count(n):
    total = 0
    for _ in range(n):
        total += 1
    return total


# ---------------------------------------------------------------------------
# A tiny 1-D SGS along a drillhole line
# ---------------------------------------------------------------------------
def sgs_1d(seed, n=20_000, r=0.95):
    """Simulate n samples along a line; each sample depends on the previous one.
    Returns the fraction of samples that are ore (grade >= 0.5 % Cu)."""
    rng = np.random.default_rng(seed)          # every realization gets its own seed
    z = [rng.standard_normal()]
    for _ in range(n - 1):                     # SEQUENTIAL: needs the previous value
        z.append(r * z[-1] + (1 - r**2) ** 0.5 * rng.standard_normal())
    grade = np.exp(-0.98 + 0.6 * np.array(z))  # turn normal scores into % Cu
    return np.mean(grade >= 0.5)


# ---------------------------------------------------------------------------
# A small 2-D SGS of Cu grade
# ---------------------------------------------------------------------------
SIZE = 40                 # the block model is SIZE x SIZE blocks
BLOCK_TONNES = 25_000     # tonnes in one block (25 m x 25 m x 15 m bench x 2.7 t/m3)
RANGE = 12.0              # how far (in blocks) grades stay similar: 12 x 25 m = 300 m
NEIGHBOURS = 12           # how many nearby values each block looks at


def similarity(distance):
    """How alike two grades are at a given distance (1 = identical, 0 = unrelated)."""
    return np.exp(-3.0 * distance / RANGE)


def _simulate(rng, known_xy, known_values):
    """Visit every block in random order. Each block's value is drawn from what its
    already-known neighbours suggest (simple kriging). Because each block uses the
    blocks simulated BEFORE it, one realization must be done step by step."""
    n_known = len(known_values)
    pts = np.zeros((n_known + SIZE * SIZE, 2))        # positions of known values
    vals = np.zeros(n_known + SIZE * SIZE)            # the known values themselves
    if n_known:
        pts[:n_known] = known_xy
        vals[:n_known] = known_values
    grid = np.empty((SIZE, SIZE))
    for block in rng.permutation(SIZE * SIZE):          # random path
        i, j = divmod(block, SIZE)
        if n_known == 0:                                # very first block: nothing known yet
            v = rng.standard_normal()
        else:
            d = np.hypot(pts[:n_known, 0] - i, pts[:n_known, 1] - j)
            k = min(NEIGHBOURS, n_known)
            near = np.argpartition(d, k - 1)[:k]        # closest known blocks
            P = pts[near]
            C = similarity(np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]))
            c = similarity(d[near])
            w = np.linalg.solve(C + 1e-9 * np.eye(k), c)   # kriging weights
            mean = w @ vals[near]
            var = max(1.0 - w @ c, 0.0)
            v = mean + np.sqrt(var) * rng.standard_normal()   # random draw
        grid[i, j] = v
        pts[n_known] = (i, j)                           # this block is now known
        vals[n_known] = v
        n_known += 1
    return grid


_DRILLHOLES = None


def drillholes():
    """30 made-up drillhole samples (positions and normal-score values)."""
    global _DRILLHOLES
    if _DRILLHOLES is None:
        rng = np.random.default_rng(2026)
        truth = _simulate(rng, [], [])
        picks = rng.choice(SIZE * SIZE, size=30, replace=False)
        xy = [divmod(int(p), SIZE) for p in picks]
        _DRILLHOLES = (xy, [truth[i, j] for i, j in xy])
    return _DRILLHOLES


def sgs_2d(seed):
    """One realization of Cu grade (%) on the 40 x 40 block model, honouring the drillholes."""
    xy, values = drillholes()
    rng = np.random.default_rng(seed)
    z = _simulate(rng, xy, values)
    return np.exp(np.log(0.45) - 0.18 + 0.6 * z)        # mean grade about 0.45 % Cu


def ore_summary(seed, cutoff=0.5):
    """Simulate one realization and return ore tonnes (Mt) and contained Cu (kt) above the cutoff."""
    grade = sgs_2d(seed)
    ore = grade >= cutoff
    ore_mt = ore.sum() * BLOCK_TONNES / 1e6
    cu_kt = (grade[ore].sum() / 100) * BLOCK_TONNES / 1e3
    return {"seed": seed, "ore_Mt": ore_mt, "cu_kt": cu_kt}


# ---------------------------------------------------------------------------
# A toy mine simulator (the "DRS" part)
# ---------------------------------------------------------------------------
# One period = one week. Values are realistic in order of magnitude for a small-to-mid open-pit
# copper operation (about 5 Mt/y through the plant), but this is still a teaching model.
PRICE = 9_500            # $ per tonne of Cu
MINING_COST = 2.5        # $ per tonne mined (ore and waste)
PLANT_COST = 12          # $ per tonne processed
REHANDLE_COST = 1.5      # $ per tonne taken back from the stockpile
RECOVERY = 0.88          # share of Cu recovered from fresh ore
STOCKPILE_RECOVERY = 0.80  # stockpiled ore oxidises and recovers less
FIXED_COST = 1.0e6       # $ per week while the operation runs (G&A, site services)
BLOCKS_MINED = 8         # blocks mined per week   (8 x 25 kt = 200 kt/week, about 10 Mt/y)
PLANT_SIZE = 4           # blocks the plant can process per week (100 kt/week, about 5 Mt/y): the bottleneck
DISCOUNT = 1.08 ** (1 / 52) - 1   # 8 % per year, as a weekly rate


def mine_npv(grade, cutoff, stockpile_cutoff, seed=0):
    """Mine the block model row by row and return the NPV in $M.

    Policy (the two numbers we want to optimize):
      grade >= cutoff                      -> straight to the plant
      stockpile_cutoff <= grade < cutoff   -> low-grade stockpile, processed when the plant has room
      grade < stockpile_cutoff             -> waste
    Each period (one week) the plant works at a random availability (breakdowns, maintenance).
    """
    rng = np.random.default_rng(seed)
    stockpile_cutoff = min(stockpile_cutoff, cutoff)
    blocks = list(grade.ravel())            # the mining sequence
    stockpile = []
    npv, period = 0.0, 0

    while blocks or stockpile:
        mined = blocks[:BLOCKS_MINED]
        blocks = blocks[BLOCKS_MINED:]

        availability = np.clip(rng.normal(1.0, 0.15), 0.4, 1.2)
        room = round(PLANT_SIZE * availability)

        to_plant = [g for g in mined if g >= cutoff]
        stockpile += [g for g in mined if stockpile_cutoff <= g < cutoff]
        stockpile += to_plant[room:]        # plant full: the rest waits on the stockpile
        to_plant = to_plant[:room]

        stockpile.sort()
        from_stockpile = []
        while len(to_plant) + len(from_stockpile) < room and stockpile:
            from_stockpile.append(stockpile.pop())     # best stockpiled block first

        cu = (sum(to_plant) * RECOVERY + sum(from_stockpile) * STOCKPILE_RECOVERY) / 100 * BLOCK_TONNES
        cash = (cu * PRICE
                - (len(to_plant) + len(from_stockpile)) * BLOCK_TONNES * PLANT_COST
                - len(from_stockpile) * BLOCK_TONNES * REHANDLE_COST
                - len(mined) * BLOCK_TONNES * MINING_COST
                - FIXED_COST)
        npv += cash / (1 + DISCOUNT) ** period
        period += 1
    return npv / 1e6
