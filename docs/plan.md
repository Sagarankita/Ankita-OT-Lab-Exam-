# Execution plan and completed quality audit

The original 55-minute plan below was completed. The user then authorized extra time to reread the question, strengthen the quantum implementation, and improve the submission. The enhanced implementation and verification are complete; see the final status at the end.

Goal: choose exactly 3 of L1–L8 to maximize ML-predicted damage within 10 inspection hours. Connect regression, gate-based quantum optimization, and parallel exhaustive verification. Exact inputs and implementation requirements are in `prompts.md`.

| Minutes | Action | Completion check |
|---|---|---|
| 0–5 | Inspect workspace instructions, Python, and installed NumPy/SciPy, regression, parallel, quantum, and PDF tools. | Choose the smallest viable stack; establish quantum fallback immediately. |
| 5–12 | Enter exact seeded data and candidates; fit/evaluate regression, refit all records, predict candidates. | Print coefficients, validation metric, and 8 predictions. |
| 12–18 | Define binary optimization; compute greedy baseline and enumerate triples. | Exactly 56 triples; print feasible count and baseline feasibility. |
| 18–28 | Evaluate all triples in parallel and measure elapsed time. | Best feasible selection, damage, and hours independently recompute correctly. |
| 28–43 | Build and validate 12-bit slack QUBO; execute shallow QAOA. | Actual simulator output, documented settings, decoded samples, feasibility, encoding check. |
| 43–51 | Build comparison and one-page report; verify references. | All required sections present; PDF rendered and visually checked. |
| 51–55 | Run full script from scratch; save output and reconcile report. | Deliverables exist, results agree, reproducible command works. |
| 55–60 | Submission buffer. | Open final files and submit. |

## Keep implementation focused

- One script, one saved output, one-page report, minimal dependencies.
- Complete the classical reference early so quantum results can be checked.
- Bound quantum optimization by evaluation count and wall-clock time. Reduce restarts or depth if needed; report settings and observed results honestly.
- If a quantum framework is unavailable, use explicit statevector gates with NumPy rather than spending the time budget installing a large stack.
- No extra UI, plotting, extensive tuning, or unnecessary tests. Check numerical correctness, feasibility, reproducibility, and report layout.

## Coverage against the question's marking scheme

| Requirement | Marks | Evidence to include |
|---|---:|---|
| Formulation | 1 | Binary variables, maximization, both constraints, 56 total triples, computed feasible count. |
| Working solution connecting all three domains | 2.5 | Regression predictions; explicit QUBO and executed gate-based quantum simulation; actual parallel exhaustive verification; greedy top-three baseline. |
| Results table | 1 | Greedy versus quantum, with exhaustive optimum as reference; locations, total predicted damage, total hours, time-limit status, runtime. |
| Interpretation | 0.5 | Whether quantum matches exhaustive search, two limitations, references. |

The generator must match exactly; regression method, quantum framework, and parallel backend are implementation choices. The question does not require hardware execution or a speedup claim. A NumPy circuit simulator is a fallback; a familiar installed quantum framework is preferable for examiner review.

## Quality priorities within the deadline

1. Preserve every marked requirement and produce actual runnable outputs.
2. Include rigorous formulation, validation metrics, explicit QUBO coefficients, and trustworthy quantum sampling evidence as detailed in `prompts.md`.
3. Save `qubo.json` and print concise diagnostics; keep the report to one readable page.
4. Spend spare time on quantum optimization only after the report pipeline works. Do not sacrifice submission files to chase a perfect quantum result.

## Final checklist

- [x] Training generator matches the question exactly; candidate feature order is correct.
- [x] Regression predicts all eight locations; fitted coefficients are plausible.
- [x] Formulation includes binary variables, objective, exact cardinality, and time limit.
- [x] Total triple count is 56; feasible count is computed, not guessed.
- [x] Greedy top-three baseline is reported even if infeasible.
- [x] All triples are evaluated in parallel; exact feasible optimum is recorded.
- [x] QUBO includes slack and penalties; its optimum is checked independently.
- [x] Actual gate-based quantum simulation runs; samples and failures are honestly reported.
- [x] Table includes selections, damage, hours, feasibility, and scoped runtimes.
- [x] Interpretation compares quantum with exhaustive optimum and states two limitations.
- [x] References are verified; all reported numbers come from the saved run.
- [x] `solution.py`, `requirements.txt`, `output.txt`, `results.md`, and one-page `results.pdf` are ready.

## Handoff

To start in another chat, use: **“Read `prompts.md` and `plan.md` in this workspace and execute the assignment now.”**

If interrupted, append a brief status here: completed files, exact next action, blocker, and remaining time. Resume from those files instead of repeating finished work.

## Completed implementation

All deliverables were generated and rerun successfully. Verified 56 triples, 35 feasible selections, all 4096 QUBO assignments, and 4096 quantum shots. Best feasible quantum sample and exhaustive search select L1,L2,L5 (40.244750 predicted damage, 10 hours). The latest one-page PDF was rendered and visually inspected. No execution blocker remains. See README.md for run instructions.

## Enhanced audit completed

- [x] Reread the original photograph and checked every input and marking requirement.
- [x] Increased QAOA to depths 1,2,3,4,6 with three seeded starts at each depth.
- [x] Exported an elementary H/Rz/CX/Rx OpenQASM circuit and independently simulated it before sampling; state fidelity agrees within 1e-10.
- [x] Achieved approximately 36.85% exact feasible probability; sampled the optimum 29 times in 4096 shots.
- [x] Disclosed the distinction between best feasible sample, modal full state (L1,L2,L8, feasible but suboptimal), and slack-marginalized measured mode (L7,L8, infeasible).
- [x] Added the candidate input/prediction table and modal quantum row while retaining one PDF page.
- [x] Added `audit.md` with rubric traceability, QUBO derivation, penalty proof, and interpretation.
- [x] Added `verify.py`; all five independent test groups pass, including exported circuit and shot reconciliation.
- [x] Updated README, prompts, diagnostics, and the organized repository.

## Repository layout

Code remains at the repository root; generated artifacts are in `results/`; planning and audit documents are in `docs/`. The submission ZIP and Python cache were removed at the user's request.
