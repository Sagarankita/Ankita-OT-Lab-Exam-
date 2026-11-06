"""Greedy selection, parallel exhaustive search, and feasibility helpers."""
import multiprocessing as mp
import os
from itertools import combinations
from time import perf_counter

import numpy as np

from .data import TIMES


def evaluate(task):
    """Top-level picklable function: exactly one evaluation per triple."""
    triple, predictions = task
    return {'indices': list(triple), 'damage': float(sum(predictions[i] for i in triple)),
            'hours': int(sum(TIMES[i] for i in triple)), 'worker_pid': os.getpid()}


def feasible(row):
    return len(row['indices']) == 3 and row['hours'] <= 10


def best(rows):
    # Treat scores within 1e-10 as ties, resolving by lexicographic location order.
    valid = [r for r in rows if feasible(r)]
    if not valid:
        return None
    score = max(r['damage'] for r in valid)
    return min((r for r in valid if abs(r['damage']-score) <= 1e-10),
               key=lambda r: tuple(r['indices']))


def names(row):
    if row is None:
        return 'Unavailable'
    return ','.join(f'L{i+1}' for i in row['indices']) or 'None (0 selected)'


def solve_classical(predictions, workers=2):
    """Evaluate all triples in process workers and check against a sequential pass."""
    t=perf_counter()
    greedy=evaluate((tuple(sorted(sorted(range(8),key=lambda i:(-predictions[i],i))[:3])),predictions))
    greedy.update(method='Greedy top-3',seconds=perf_counter()-t)
    triples=list(combinations(range(8),3)); assert len(triples)==56
    tasks=[(triple,predictions.tolist()) for triple in triples]
    t=perf_counter()
    with mp.get_context('spawn').Pool(workers) as pool:
        evaluated=pool.map(evaluate,tasks,chunksize=1)
    optimum=best(evaluated)
    parallel_seconds=perf_counter()-t
    assert len(evaluated)==56 and len({tuple(r['indices']) for r in evaluated})==56
    sequential=[evaluate(task) for task in tasks]
    assert all(a['indices']==b['indices'] and a['hours']==b['hours'] and np.isclose(a['damage'],b['damage']) for a,b in zip(evaluated,sequential))
    assert best(sequential)['indices']==optimum['indices']
    optimum.update(method='Parallel exhaustive',seconds=parallel_seconds)
    feasible_count=sum(feasible(r) for r in evaluated)

    return dict(greedy=greedy, optimum=optimum, evaluated=evaluated, feasible_count=feasible_count)
