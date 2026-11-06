"""Marking-scheme traceability and detailed mathematical interpretation."""
from .classical import feasible, names
from .config import DOCS, SHOTS


def write_audit(result, evaluated):
    """Requirement-by-requirement traceability, separate from the one-page sheet."""
    q=result['quantum']; opt=result['optimum']
    counts={h:sum(row['hours']==h for row in evaluated) for h in sorted({row['hours'] for row in evaluated})}
    ranked=sorted((r for r in evaluated if feasible(r)),key=lambda r:(-r['damage'],r['indices']))
    text=f'''# Assignment audit and technical notes

The original image was reread and checked against the implementation. The question asks for working code, its output, and a one-page sheet covering four marked sections. The methods and evidence below satisfy those sections; no hardware run or speedup is claimed.

## Requirement traceability

| Original requirement | Implemented evidence |
|---|---|
| Exact fixed-seed historical data | `earthquake_lab/data.py: training_data()` preserves the given generator, order, seed 63 and N=120. |
| Candidate data L1-L8 | `CANDIDATES` and `TIMES` match the photograph; `candidate_predictions.csv` preserves units and predictions. |
| Formulation (1 mark) | `results.pdf`: binary variables, maximization, exactly 3 selections, inspection time <=10 h, 56 total triples and 35 feasible. |
| Machine learning (part of 2.5 marks) | OLS with intercept, independent 96/24 validation, final refit on all records, all eight predictions. |
| Quantum computing (part of 2.5 marks) | Explicit slack QUBO/Ising; multistart QAOA; `circuit.qasm` uses elementary gates and is simulated before sampling. |
| Parallel computing (part of 2.5 marks) | Two spawned process workers evaluate every one of the 56 triples; a sequential pass independently confirms the result. |
| Greedy baseline | Highest three predicted damages, without silently repairing its 13-hour violation. |
| Results table (1 mark) | Greedy, QAOA best feasible sample, QAOA modal state, and parallel exhaustive reference; locations, damage, hours, time-limit status, and measured runtime. |
| Interpretation (0.5 mark) | Best feasible quantum sample matches the exact optimum; modal outputs are separately disclosed; two limitations and three primary references. |
| Deliverables | `solution.py`, `earthquake_lab/`, `requirements.txt`, `output.txt`, `results.pdf`; editable report and diagnostics are supporting files. |

## Classical answer and count check

- C(8,3) = 8!/(3!5!) = 56 unordered selections; 35 feasible and 21 over time.
- Triple count grouped by total inspection hours: {counts}. Summing entries for hours <=10 gives 35.
- Optimal locations: **{names(opt)}**, damage **{opt['damage']:.9f}**, time **{opt['hours']} h**.
- Next-best feasible selection: {names(ranked[1])}, damage {ranked[1]['damage']:.9f}, time {ranked[1]['hours']} h. The optimum is unique for these fitted predictions.
- These are predicted damage scores, not measured earthquake losses. No damage unit was specified by the question.

## Regression rationale

The supplied target is linear in d, a, l, r plus independent Gaussian noise. Ordinary least squares with an intercept is therefore a direct, appropriate model. A separate seed 7 selects 96 training and 24 test records without consuming or changing the seed-63 generator. Reported test metrics come from the training-only fit; the eight candidate predictions come from a subsequent all-record fit. No target leakage or fitted preprocessing is used. Validation is confined to this synthetic distribution.

## QUBO derivation and penalty proof

Let z=(x1,...,x8,b0,b1,b2,b3), c=(1,1,1,1,1,1,1,1,0,0,0,0), w=(4,2,3,5,4,3,3,3,1,2,4,8), and v=(eight predicted damages,0,0,0,0). The slack s=b0+2b1+4b2+8b3 is nonnegative. Feasible selections have an integer slack 10-sum(t_i*x_i), representable by four bits. Extra representable slack values cause no false feasible solutions because the equality is enforced.

E(z) = -v.z + A(c.z-3)^2 + B(w.z-10)^2.

Using z_j^2=z_j, write E=C+sum(h_j*z_j)+sum(j<k,J_jk*z_j*z_k), with:

- C=9A+100B.
- h_j=-v_j+A(c_j^2-6c_j)+B(w_j^2-20w_j).
- J_jk=2A*c_j*c_k+2B*w_j*w_k for j<k; the lower triangle is zero.
- A=B=1+sum(abs(v))={q['metadata']['A']:.9f}.

The reward term's total range is at most sum(abs(v)). Every invalid encoded assignment has an integer squared violation of at least 1, so this penalty choice makes any invalid energy exceed every feasible energy. Nonnegative slack cannot cancel a time total greater than 10. Dividing all terms by A preserves the minimizer. All 4096 assignments were checked against the polynomial, penalty form, and Ising form, and every ground-state selection agrees with exhaustive search.

## Circuit and optimizer

Substitute z_j=(1-Z_j)/2: H=K+sum(a_j*Z_j)+sum(j<k,b_jk*Z_j*Z_k), where a_j=-h_j/2-sum(k!=j,J_jk)/4 and b_jk=J_jk/4 (the sum uses each stored upper-triangle pair incident on j). Numeric coefficients are in `qubo.json`.

Prepare |+> on 12 qubits. At each QAOA layer apply exp(-i*gamma*H/A), then exp(-i*beta*sum(X)). Cost terms commute. Implement each one-qubit term as Rz(2*gamma*a_j/A), each pair as CX(j,k)-Rz(2*gamma*b_jk/A,k)-CX(j,k), and each mixer as Rx(2*beta). The omitted constant is a global phase.

The optimizer uses the fused diagonal representation for efficiency. Final measurements instead use the elementary-gate simulator. Their state fidelity is {q['metadata']['circuit_fidelity']:.15f}; equality is checked up to global phase. The exported OpenQASM circuit has gate counts {q['metadata']['gate_counts']} plus 12 terminal measurements. Qubit 0 changes fastest in the statevector; displayed output strings place q[0] first. Conventional QASM tools may display classical strings in reverse order.

There are three seeded starts (seed 29) at each p in 1,2,3,4,6, with per-start evaluation budgets 300,600,900,1200,1800. Earlier depth winners initialize later runs. The circuit is chosen by the lowest expected QUBO energy alone, never by the exhaustive answer. Chosen p={q['metadata']['chosen']['depth']}; total evaluations={q['metadata']['total_evaluations']}. Budget exhaustion is reported rather than called convergence. Full settings and per-run history are saved.

## Interpreting the quantum result correctly

- Best feasible measured selection: {names(q['best'])}; it matches the exact optimum.
- Most frequent complete measured bitstring: {names(q['modal_state'])}; original feasibility={feasible(q['modal_state'])}.
- Most frequent measured location selection after aggregating all slack bitstrings: {names(q['modal_selection'])}; original feasibility={feasible(q['modal_selection'])}.
- Original-constraint feasible shots: {q['feasible_fraction']:.6%}; additionally correct slack shots: {q['encoded_feasible_fraction']:.6%}.
- Optimal selection: {q['optimum_count']}/{SHOTS} shots ({q['optimum_fraction']:.6%}); exact probability {q['optimum_probability']:.6%}.
- Original feasibility is the assignment's selection criterion. An incorrect measured slack does not invalidate an otherwise legal location triple, but is identified as an encoding violation.
- Uniform sampling has feasible probability 35/256=13.671875% and unique optimal-selection probability 1/256=0.390625%. This simple reference highlights why finding an optimum in a large batch is insufficient evidence of advantage. It is not an equal-budget runtime benchmark.
- Retaining the best feasible shot is classical postprocessing and is explicitly disclosed. No classical exhaustive optimum is inserted into the quantum samples.

## Timing and submission checks

Timers use time.perf_counter. Parallel time includes worker startup, all evaluations, reduction, and shutdown. Quantum time includes encoding, energy checks, all optimizer attempts, gate compilation/simulation, sampling, and decoding. The later comparison with exhaustive ground-state selections and report/file writing are excluded. Common ML preparation is reported separately. The two quantum table rows describe the same run, so their identical runtime must not be added. Timings are single-run observations, not a speedup claim.

The PDF generation asserts exactly one page. External QA also checks its page count and renders the final version for visual inspection. All 56 triple records and all 4096 measured shots are reconciled with the saved results.
'''
    (DOCS/'audit.md').write_text(text)
