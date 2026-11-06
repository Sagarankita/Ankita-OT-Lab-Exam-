"""Orchestrate the three domains without implementing their algorithms here."""
import numpy as np

from .artifacts import save_artifacts
from .audit import write_audit
from .classical import evaluate, feasible, solve_classical
from .config import DOCS, OUT, SHOTS
from .prediction import fit_predict
from .quantum.qaoa import run_quantum
from .reporting import create_reports, print_predictions, print_results


def run():
    """Run the complete experiment and generate the submission artifacts."""
    OUT.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    prediction = fit_predict()
    print_predictions(prediction)
    predictions = np.array(prediction['predictions'])
    classical = solve_classical(predictions)
    greedy = classical['greedy']
    optimum = classical['optimum']
    evaluated = classical['evaluated']
    feasible_count = classical['feasible_count']
    print(f'\nAll triples: 56; feasible triples: {feasible_count}')
    print('Parallel backend: multiprocessing.Pool, spawn, 2 workers; actual task PIDs:',
          sorted({row['worker_pid'] for row in evaluated}))
    print('Running gate-based QAOA statevector simulation...', flush=True)
    q = run_quantum(predictions)
    # Validate encoding independently of QAOA; never feed this optimum into QAOA.
    ground_indices=np.flatnonzero(np.isclose(q['energies'],q['energies'].min(),atol=1e-8,rtol=0))
    for k in ground_indices:
        row=evaluate((tuple(np.flatnonzero(q['bits'][k,:8]).tolist()),predictions))
        assert feasible(row) and np.isclose(row['damage'],optimum['damage'])
    optimum_id=sum(1<<i for i in optimum['indices'])
    q['optimum_fraction']=float(q['selection_counts'][optimum_id]/SHOTS)
    q['optimum_count']=int(q['selection_counts'][optimum_id])
    q['optimum_probability']=float(q['selection_probs'][optimum_id])
    q['metadata']['checks']='all 4096 polynomial/penalty/Ising energies agree; QUBO minimum agrees with exhaustive optimum; elementary-gate and fused simulators agree; probabilities and shots normalized'

    comparison=[greedy]
    if q['best']:
        q['best'].update(method='QAOA best feasible',seconds=q['seconds']); comparison.append(q['best'])
    q['modal_state'].update(method='QAOA modal state',seconds=q['seconds'])
    comparison.append(q['modal_state'])
    comparison.append(optimum)
    result = dict(prediction, feasible_count=feasible_count, optimum=optimum,
                  quantum=q, comparison=comparison)
    print_results(result)
    save_artifacts(result, evaluated)
    create_reports(result)
    write_audit(result, evaluated)
    print('\nSaved results and diagnostics in results/; requirement audit in docs/audit.md.')
    return result
