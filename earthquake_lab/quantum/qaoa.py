"""Bounded multistart QAOA optimization and seeded circuit measurements."""
from time import perf_counter

import numpy as np

from ..classical import evaluate, best
from ..config import OPTIMIZATION_SCHEDULE, RESTARTS, SHOTS
from ..data import TIMES
from .encoding import qubo
from .circuit import statevector, gate_circuit, simulate_gates, qasm_text


def nelder_mead(fun, initial, steps, budget=150):
    """Small deterministic Nelder-Mead optimizer; fixed function-evaluation cap."""
    initial = np.array(initial, dtype=float)
    simplex = np.vstack([initial, initial + np.diag(steps)])
    values = np.array([fun(x) for x in simplex]); calls = len(values)
    converged = False
    while calls < budget:
        order = np.argsort(values); simplex, values = simplex[order], values[order]
        if np.ptp(values)<1e-9 and np.max(abs(simplex-simplex[0]))<1e-6:
            converged = True; break
        center = simplex[:-1].mean(axis=0)
        reflected = center + (center-simplex[-1])
        fr = fun(reflected); calls += 1
        if fr < values[0] and calls < budget:
            expanded = center + 2*(reflected-center)
            fe = fun(expanded); calls += 1
            simplex[-1], values[-1] = (expanded,fe) if fe < fr else (reflected,fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected,fr
        elif calls < budget:
            outside = fr < values[-1]
            contracted = center + .5*((reflected if outside else simplex[-1])-center)
            fc = fun(contracted); calls += 1
            if fc < (fr if outside else values[-1]):
                simplex[-1], values[-1] = contracted,fc
            else:
                for j in range(1,len(simplex)):
                    if calls >= budget: break
                    simplex[j] = simplex[0] + .5*(simplex[j]-simplex[0])
                    values[j] = fun(simplex[j]); calls += 1
    i = np.argmin(values)
    return simplex[i], float(values[i]), calls, converged


def run_quantum(predictions):
    start = perf_counter()
    energies, bits, metadata = qubo(predictions)
    # Scale the entire energy by a positive constant; remove a global phase.
    cost = (energies-metadata['constant']) / metadata['energy_scale']
    attempts = []; previous = None
    initialization_rng = np.random.default_rng(29)
    original_feasible = (bits[:,:8].sum(axis=1)==3) & (bits[:,:8] @ TIMES <= 10)
    for p,budget in OPTIMIZATION_SCHEDULE:
        depth_attempts = []
        for restart in range(RESTARTS):
            if previous is None:
                initial = ([.012,.3] if restart==0 else
                           [.006+initialization_rng.random()*.03,-.1-initialization_rng.random()*.6])
            else:
                initial = list(previous)+[.006,-.1]*(p-len(previous)//2)
                if restart:
                    initial = np.array(initial)+np.tile([.003,.08],p)*initialization_rng.normal(size=2*p)
            def objective(theta):
                state = statevector(theta,cost)
                return float(np.dot(abs(state)**2,cost))
            theta,value,calls,converged = nelder_mead(objective,initial,np.tile([.003,.08],p),budget)
            probability = abs(statevector(theta,cost))**2
            attempt = dict(depth=p,restart=restart,parameters=theta.tolist(),scaled_expectation=value,
                           evaluations=calls,budget=budget,converged=converged,
                           feasible_probability=float(probability[original_feasible].sum()))
            attempts.append(attempt); depth_attempts.append(attempt)
        depth_best = min(depth_attempts,key=lambda a:a['scaled_expectation'])
        previous = depth_best['parameters']
        print(f"  p={p}: best scaled expectation={depth_best['scaled_expectation']:.6f}; "
              f"feasible probability={depth_best['feasible_probability']:.2%}",flush=True)
    chosen = min(attempts,key=lambda a:a['scaled_expectation'])
    reference_state = statevector(chosen['parameters'],cost)
    gates = gate_circuit(chosen['parameters'],metadata)
    state = simulate_gates(gates)
    fidelity = float(abs(np.vdot(reference_state,state))**2)
    assert np.isclose(fidelity,1,atol=1e-10,rtol=0)
    probs = abs(state)**2
    assert np.isclose(sum(probs),1,atol=1e-12)
    samples = np.random.default_rng(2026).choice(4096,size=SHOTS,p=probs/probs.sum())
    counts = np.bincount(samples,minlength=4096)
    assert int(counts.sum()) == SHOTS
    observed = []
    for k in np.flatnonzero(counts):
        row = evaluate((tuple(np.flatnonzero(bits[k,:8]).tolist()),predictions))
        row.update(state=int(k), count=int(counts[k]), slack=int(bits[k,8:] @ [1,2,4,8]),
                   encoded_feasible=bool(bits[k,:8].sum()==3 and bits[k] @ np.r_[TIMES,[1,2,4,8]]==10))
        observed.append(row)
    sampled_best = best(observed)
    modal_state = max(observed,key=lambda r:(r['count'],-r['state']))
    # Marginalize all 16 slack assignments for each 8-bit location selection.
    selection_probs = np.bincount(np.arange(4096)&255,weights=probs,minlength=256)
    selection_counts = np.bincount(samples&255,minlength=256)
    modal_selection_id = int(np.argmax(selection_counts))
    modal_selection = evaluate((tuple(i for i in range(8) if modal_selection_id >> i & 1),predictions))
    feasible_mask = (bits[:,:8].sum(axis=1)==3) & (bits[:,:8] @ TIMES <= 10)
    encoded_mask = (bits[:,:8].sum(axis=1)==3) & (bits @ np.r_[TIMES,[1,2,4,8]]==10)
    metadata.update(attempts=attempts, chosen=chosen, shots=SHOTS, sampling_seed=2026,
                    simulator='NumPy exact gate-based statevector', optimizer='bounded Nelder-Mead',
                    training_seed=63, split_seed=7,initialization_seed=29,
                    gate_counts={g:sum(x[0]==g for x in gates) for g in ['h','rz','rx','cx']},
                    circuit_fidelity=fidelity,total_evaluations=sum(a['evaluations'] for a in attempts))
    elapsed = perf_counter()-start
    return dict(best=sampled_best, modal_state=modal_state, modal_selection=modal_selection,
                feasible_fraction=float(counts[feasible_mask].sum()/SHOTS),
                encoded_feasible_fraction=float(counts[encoded_mask].sum()/SHOTS),
                feasible_probability=float(probs[feasible_mask].sum()),
                seconds=elapsed, metadata=metadata, counts=counts, probabilities=probs,
                selection_counts=selection_counts, selection_probs=selection_probs,
                energies=energies, bits=bits, qasm=qasm_text(gates))
