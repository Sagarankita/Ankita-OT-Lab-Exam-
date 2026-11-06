"""Exact examination inputs and the unchanged fixed-seed data generator."""
import numpy as np


CANDIDATES = np.array([[14,35,.8,6], [9,50,.3,12], [18,20,.6,25],
                       [6,55,.9,8], [16,45,.5,10], [11,30,.7,18],
                       [4,40,.2,30], [12,25,.4,15]], dtype=float)
TIMES = np.array([4,2,3,5,4,3,3,3], dtype=int)


def training_data():
    # These generator statements match the examination question exactly.
    import numpy as np
    rng = np.random.default_rng(63); N = 120
    d = rng.uniform(1, 20, N); a = rng.uniform(5, 60, N)
    l = rng.uniform(0, 1, N); r = rng.uniform(2, 40, N)
    damage = 0.30*d + 0.08*a + 4.0*l - 0.12*r + 5 + rng.normal(0, 0.5, N)
    return np.column_stack([d, a, l, r]), damage
