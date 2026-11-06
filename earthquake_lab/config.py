"""Shared paths, reproducibility settings, and primary references."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DOCS = ROOT / "docs"
SHOTS = 4096
OPTIMIZATION_SCHEDULE = [(1,300), (2,600), (3,900), (4,1200), (6,1800)]
RESTARTS = 3
REFERENCES = [
 ('NumPy: least-squares regression', 'https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html'),
 ('Farhi, Goldstone & Gutmann (2014), QAOA', 'https://arxiv.org/abs/1411.4028'),
 ('Python: multiprocessing process pools', 'https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers'),
]
