# Earthquake Inspection Optimization

Select exactly three of eight locations to maximize predicted earthquake damage, with a total inspection time of at most 10 hours. This assignment connects **machine learning**, **quantum computing**, and **parallel computing**.

## Run

Use Python 3.10 or later:

```sh
python3 -m pip install -r requirements.txt
python3 solution.py
python3 verify.py
```

Run the solution as a script so spawned multiprocessing workers can import its functions. Generated files are saved in `results/`, and the requirement audit is updated in `docs/audit.md`. Tested with Python 3.12.14, NumPy 2.3.5, and ReportLab 4.4.9. A full run took about 10–20 seconds on the preparation computer; times vary.

## Method

1. Generate the exact 120 historical records using seed 63. Validate ordinary least squares on an independent 96/24 split, then refit all records to predict the eight candidates.
2. Compute the unrestricted greedy top-three baseline and report its time-limit violation.
3. Encode the constraints as a 12-qubit QUBO with nonnegative slack. Run QAOA at depths 1, 2, 3, 4, and 6 with three seeded starts each. Independently simulate the final elementary H/Rz/CX/Rx circuit before sampling 4,096 shots. No quantum hardware or cloud account is required.
4. Evaluate all 56 triples with two process workers and verify the exact optimum independently.

## Results

**L1, L2, L5** is the exact optimum: predicted damage **40.244750**, inspection time **10 hours**. Of the 56 triples, **35** satisfy the time limit. The greedy selection L1, L4, L5 takes **13 hours** and is infeasible.

QAOA sampled the optimum **29 times in 4,096 shots**. The report distinguishes the best feasible sample from the modal full bitstring and the modal location selection after combining slack outcomes. Retaining the best feasible shot is classical postprocessing; no quantum advantage is claimed.

## Files

| Path | Purpose |
|---|---|
| [solution.py](solution.py) | Complete implementation and report generation |
| [verify.py](verify.py) | Five independent verification test groups |
| [requirements.txt](requirements.txt) | Required external dependencies |
| [results/results.pdf](results/results.pdf) | Visually checked, one-page results sheet |
| [results/output.txt](results/output.txt) | Saved console output from a successful run |
| [results/results.md](results/results.md) | Editable results sheet |
| [results/circuit.qasm](results/circuit.qasm) | Executable elementary quantum circuit |
| [docs/audit.md](docs/audit.md) | Marking-scheme audit, derivations, and interpretation |

`results/` also contains machine-readable results, QUBO coefficients, candidate predictions, all 56 selections, quantum shot counts, location probabilities, and optimizer history. Planning instructions are preserved in `docs/`.

## Verification and limitations

`verify.py` checks the input table, regression predictions, every classical triple, elementary gate behavior, the exported QASM circuit, and consistency of all saved shots and marginal counts. PDF generation asserts exactly one page; the committed PDF was also rendered and visually inspected.

Quantum runtime includes encoding, energy checks, all optimizer attempts, gate simulation, sampling, and decoding. Parallel runtime includes process startup, evaluation, reduction, and shutdown. Common ML preparation is reported separately. Both quantum table rows describe the same run; their runtimes must not be added.

The data are synthetic, and QAOA convergence is not guaranteed. The local statevector simulator scales exponentially and does not model hardware noise. References and full technical details are in the report and audit.
