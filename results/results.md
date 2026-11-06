# Earthquake inspection - results

## Formulation
Binary x_i indicates inspection. Maximize sum(yhat_i*x_i), subject to sum(x_i)=3 and sum(t_i*x_i)<=10. There are C(8,3)=56 triples; 35 meet the time limit.

## Connected solution
OLS with intercept matches the linear synthetic generator (seed 63, N=120). Independent 80/20 split (seed 7): MAE 0.4137, RMSE 0.5260, R2 0.9632. Final fit uses all 120 records: yhat=5.129735+0.297040*d+0.080070*a+3.976314*l-0.125008*r.

| Location | d (thousand/km2) | a (years) | l (0-1) | r (km) | Time (h) | Predicted damage |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 14 | 35 | 0.8 | 6 | 4 | 14.5218 |
| L2 | 9 | 50 | 0.3 | 12 | 2 | 11.4994 |
| L3 | 18 | 20 | 0.6 | 25 | 3 | 11.3384 |
| L4 | 6 | 55 | 0.9 | 8 | 5 | 13.8944 |
| L5 | 16 | 45 | 0.5 | 10 | 4 | 14.2236 |
| L6 | 11 | 30 | 0.7 | 18 | 3 | 11.3326 |
| L7 | 4 | 40 | 0.2 | 30 | 3 | 6.5657 |
| L8 | 12 | 25 | 0.4 | 15 | 3 | 10.4114 |

Let s=b0+2b1+4b2+8b3. Minimize E=-sum(yhat_i*x_i)+A(sum(x_i)-3)^2+B(sum(t_i*x_i)+s-10)^2. A=B=1+sum(abs(yhat_i))=94.7873; a unit violation costs more than the reward range. Nonnegative slack prevents over-time solutions. The QUBO uses 12 qubits; all 4096 energies were checked against its expanded polynomial and Ising form. QAOA: |+>^12, alternating exp(-i*gamma*E/A) and Rx(2*beta); depths 1,2,3,4,6, three starts each, 11465 optimizer evaluations total; chosen p=6. Exact expectations train angles. The final elementary H/Rz/CX/Rx circuit is simulated independently (fidelity > 1-1e-10) and sampled for 4096 shots, seed 2026. Circuit: circuit.qasm; coefficients/settings: qubo.json.

A spawn multiprocessing Pool with 2 workers evaluates each triple once; a sequential pass confirms the optimum. Runtimes below are single-run wall times: greedy sorting/evaluation; quantum encoding, validation, optimization and sampling; parallel pool startup/evaluation/reduction. Common data/ML time is 0.005073 s and is excluded. Process overhead dominates this tiny workload.

## Measured comparison
| Method | Locations | Damage | Hours | Time limit met? | Runtime (s) |
|---|---|---:|---:|---|---:|
| Greedy top-3 | L1,L4,L5 | 42.639800 | 13 | False | 0.000007 |
| QAOA best feasible | L1,L2,L5 | 40.244750 | 10 | True | 9.812743 |
| QAOA modal state | L1,L2,L8 | 36.432519 | 9 | True | 9.812743 |
| Parallel exhaustive | L1,L2,L5 | 40.244750 | 10 | True | 0.075279 |

## Interpretation
The best feasible QAOA sample matches exhaustive search. Damage gap: 0.000000 (0.0000%). Optimum selection: 29/4096 shots (0.71%); feasible shots: 36.91%. The modal full bitstring selects L1,L2,L8; the modal selection after combining slack outcomes is L7,L8 (feasible: False). Observing the optimum among 4096 shots does not establish concentration or quantum advantage.

## Two limitations
1. Synthetic linear data and a small held-out set do not establish real earthquake accuracy; prediction uncertainty may change the selected triple. 2. Shallow QAOA and a bounded local optimizer offer no convergence guarantee; classical statevector memory grows exponentially. No hardware noise is modeled.

## References
1. [NumPy: least-squares regression](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
2. [Farhi, Goldstone & Gutmann (2014), QAOA](https://arxiv.org/abs/1411.4028)
3. [Python: multiprocessing process pools](https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers)
