# Execution prompt

Use this prompt with `plan.md` in this workspace:

> Complete the earthquake inspection assignment in `plan.md`. The initial budget was 55 minutes; the user subsequently authorized more time for a full audit and enhancements. Work autonomously; keep messages and tool output concise. Inspect the environment first, reuse installed packages, and avoid unnecessary dependencies. Implement, run, verify, and save the deliverables; do not stop at a proposed solution. Follow the exact data and requirements below. Report measured results only. Never claim quantum success unless the actual circuit run supports it. If a dependency blocks execution, try a lightweight local alternative and clearly identify any remaining gap. Prioritize a complete, reproducible submission over extra features. Do not create subagents, a website, or a presentation.

## Exact training data

Preserve these statements, order, seed, and values:

```python
import numpy as np
rng = np.random.default_rng(63); N = 120
d = rng.uniform(1, 20, N); a = rng.uniform(5, 60, N)
l = rng.uniform(0, 1, N); r = rng.uniform(2, 40, N)
damage = 0.30*d + 0.08*a + 4.0*l - 0.12*r + 5 + rng.normal(0, 0.5, N)
```

Features must remain ordered `[d, a, l, r]`:

| Location | d | a | l | r | Time (h) |
|---|---:|---:|---:|---:|---:|
| L1 | 14 | 35 | 0.8 | 6 | 4 |
| L2 | 9 | 50 | 0.3 | 12 | 2 |
| L3 | 18 | 20 | 0.6 | 25 | 3 |
| L4 | 6 | 55 | 0.9 | 8 | 5 |
| L5 | 16 | 45 | 0.5 | 10 | 4 |
| L6 | 11 | 30 | 0.7 | 18 | 3 |
| L7 | 4 | 40 | 0.2 | 30 | 3 |
| L8 | 12 | 25 | 0.4 | 15 | 3 |

## Implementation contract

- Fit ordinary linear regression with an intercept. Use a fixed train/test split for a small validation check, then refit on all 120 records for final candidate predictions. NumPy least squares is acceptable if scikit-learn is unavailable.
- Define binary `x_i`: maximize `sum(pred_i*x_i)`, subject to `sum(x_i)=3` and `sum(time_i*x_i)<=10`.
- Greedy baseline means the unrestricted top three predictions. Report its feasibility; do not silently repair it.
- Enumerate all 56 triples in parallel with joblib or multiprocessing; count feasible triples and find the exact feasible optimum. Use a main guard for multiprocessing. Include process startup in its stated runtime; small workloads may be slower in parallel.
- Quantum formulation: minimize `-sum(pred_i*x_i) + A*(sum(x_i)-3)^2 + B*(sum(time_i*x_i)+s-10)^2`, with nonnegative integer slack `s=b0+2*b1+4*b2+8*b3`. This uses 12 qubits. Explain penalty selection; a conservative choice is `A=B=1+sum(abs(pred_i))` before optional positive global rescaling.
- Execute actual gate-based QAOA on a simulator. Prefer an installed framework; if unavailable, a NumPy statevector simulator with explicit Hadamard, diagonal cost, and Rx mixer gates is acceptable. Use fixed seeds, shallow depth, and a bounded optimizer. State initialization, depth, shots, optimizer budget, and bit ordering. Avoid dense 4096-by-4096 matrices.
- Decode sampled bitstrings consistently. Report the most probable measured bitstring's feasibility and the best feasible measured selection separately. Do not substitute the exhaustive optimum for a quantum sample. If no feasible sample occurs, state that.
- Verify the QUBO minimum respects the original constraints and agrees with the exact optimum (4096 assignments are cheap). This validates the encoding, not QAOA convergence.
- Compare greedy, quantum, and parallel exhaustive results: selected locations, predicted damage, hours, feasibility, and clearly scoped measured runtime. Distinguish common ML preparation from solver time.
- Include two limitations: synthetic data/generalization and simulator/heuristic scalability. Add verified references for regression, QAOA, and parallel execution. Keep claims modest.

## Required files

- `solution.py`: small runnable entry point. Keep the implementation modular in `earthquake_lab/`, with separate data, regression, classical search, quantum encoding/circuit/QAOA, reporting, and artifact modules. Preserve fixed seeds and concise explanatory comments.
- `requirements.txt`: only actual external dependencies used.
- `output.txt`: output from a successful full run, including predictions, counts, comparison, and quantum settings.
- `results.md` and one-page `results.pdf`: formulation, all three connected domains, results table, interpretation, two limitations, references.

Use the available PDF skill when producing the PDF; render and inspect it for clipping and readability. Finish with file links, run command, and any actual limitations. Read `plan.md` for priorities and checks.

## Technical depth required

Keep instructions concise, but do not simplify away the mathematics or evidence:

1. **Regression:** use an 80/20 split with a separate fixed seed (do not consume the training generator's RNG). Report held-out MAE, RMSE, and R², then print the all-record fitted equation and candidate predictions. Explain why linear regression fits the supplied generative structure. Split before any learned preprocessing; these validation metrics describe synthetic data only.
2. **QUBO derivation:** show the slack equality and expand the polynomial into an explicit coefficient convention. Let `z=(x1,...,x8,b0,...,b3)`, `c=(1,...,1,0,0,0,0)`, `w=(4,2,3,5,4,3,3,3,1,2,4,8)`, and `v=(pred1,...,pred8,0,0,0,0)`. For `E=C+sum(h_j*z_j)+sum_{j<k}(J_jk*z_j*z_k)`, use `C=9*A+100*B`, `h_j=-v_j+A*(c_j*c_j-6*c_j)+B*(w_j*w_j-20*w_j)`, and `J_jk=2*A*c_j*c_k+2*B*w_j*w_k`. Do not double-count off-diagonal terms. Save coefficients/settings in `qubo.json` for inspection. Ising conversion is optional because the question permits QUBO.
3. **Penalty justification:** squared violations are integers and any infeasible assignment incurs at least one penalty. With `A=B=1+sum(abs(pred_i))`, that exceeds the full possible range of the reward term. Explain why slack cannot make an over-time selection feasible. Global rescaling must divide the entire energy by one positive constant.
4. **Circuit:** document `|+>^12`, cost unitary `exp(-i*gamma*H_C)`, and mixer `exp(-i*beta*sum(X))`, implemented with `Rx(2*beta)`. Statevector diagonal evolution must use the derived QUBO energies. Use exact expectation for optimization and actual seeded shot sampling for reported measured outcomes; disclose that distinction. Enhanced schedule: depths 1,2,3,4,6 with three starts per depth and budgets 300,600,900,1200,1800 evaluations per start; 4096 final shots. Report actual settings, convergence status, and objective evaluations. Compile/export the chosen circuit to H/Rz/CX/Rx gates, simulate those elementary gates independently, verify agreement up to global phase, and use that explicit circuit state for sampling.
5. **Quantum evidence:** report original-constraint feasible shot fraction, optimum-selection shot fraction, and absolute/percentage damage gap to exhaustive search for the best feasible sample. Aggregate probabilities over slack bits for location-selection probabilities. State whether matching the optimum means it was observed or was the modal selection. A best-of-many sample match alone is weak evidence; shallow circuits may be close to random sampling. Do not claim quantum advantage.
6. **Parallel evidence:** use actual process workers (for example, 2), report backend and worker count, and check every triple is evaluated once. Validate the parallel answer against a small sequential pass. Measure with `time.perf_counter`; benchmark once per method for this deadline and acknowledge single-run runtime variability. Resolve equal scores deterministically and use tolerance for floating-point comparisons.
7. **Correctness checks:** QUBO-polynomial and penalty-form energies agree on all 4096 assignments; their minimum maps to an exact optimal feasible triple; state probabilities sum to one; sampled counts sum to shots; decoded totals agree with direct sums. Keep these as lightweight assertions rather than a large test suite.

The one-page report should contain the essential equations, metrics, and conclusions. Put expanded coefficients, detailed settings, and diagnostic tables in code/output so the page remains readable. `qubo.json` is supporting material, not a replacement for working quantum code.

## Enhanced audit deliverables

Keep `results.pdf` exactly one page and show candidate predictions clearly. In addition to the primary deliverables, save `circuit.qasm`, `candidate_predictions.csv`, all combination/shot/marginal/training diagnostics, `audit.md` with marking-scheme traceability and derivations, and `verify.py` with meaningful simulator/export/output consistency checks. Disclose both modal full-bitstring and slack-marginalized location outcomes. Keep the verified final files in the repository: code at root, outputs in `results/`, documents in `docs/`. Do not create a ZIP archive.
