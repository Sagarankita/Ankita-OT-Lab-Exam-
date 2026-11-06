"""CSV, JSON, and circuit artifacts for independent inspection."""
import csv
import json

import numpy as np

from .classical import evaluate, feasible, names
from .config import OUT
from .data import CANDIDATES, TIMES


def save_artifacts(result, evaluated):
    """Write numerical diagnostics separately from presentation code."""
    q = result['quantum']
    predictions = np.asarray(result['predictions'])
    (OUT/'qubo.json').write_text(json.dumps(q['metadata'],indent=2)+'\n')
    (OUT/'circuit.qasm').write_text(q['qasm'])
    # Save all triples and observed quantum counts for independent inspection.
    with (OUT/'all_selections.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['locations','predicted_damage','hours','feasible'])
        for row in evaluated:writer.writerow([names(row),row['damage'],row['hours'],feasible(row)])
    with (OUT/'candidate_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['location','density_thousand_per_km2','age_years','liquefaction_index','distance_km','inspection_hours','predicted_damage'])
        for i,(features,pred,hours) in enumerate(zip(CANDIDATES,predictions,TIMES),1):
            writer.writerow([f'L{i}',*features,int(hours),float(pred)])
    with (OUT/'quantum_training.csv').open('w',newline='') as f:
        fields=['depth','restart','scaled_expectation','evaluations','budget','converged','feasible_probability']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader()
        writer.writerows(q['metadata']['attempts'])
    with (OUT/'quantum_selections.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['locations','count','probability','predicted_damage','hours','original_feasible'])
        for k in np.argsort(-q['selection_probs'],kind='stable'):
            row=evaluate((tuple(i for i in range(8) if k >> i & 1),predictions))
            writer.writerow([names(row),int(q['selection_counts'][k]),q['selection_probs'][k],row['damage'],row['hours'],feasible(row)])
    with (OUT/'quantum_samples.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator="\n");writer.writerow(['z0_to_z11','count','probability','locations','slack','hours','original_feasible','encoded_feasible'])
        for k in np.flatnonzero(q['counts']):
            b=q['bits'][k];row=evaluate((tuple(np.flatnonzero(b[:8]).tolist()),predictions))
            writer.writerow([''.join(str(int(x)) for x in b),int(q['counts'][k]),q['probabilities'][k],names(row),int(b[8:]@[1,2,4,8]),row['hours'],feasible(row),bool(b[:8].sum()==3 and b@np.r_[TIMES,[1,2,4,8]]==10)])

    summary={k:v for k,v in result.items() if k!='quantum'}
    summary['quantum']={k:v for k,v in q.items() if k not in ['counts','probabilities','selection_counts','selection_probs','energies','bits','qasm']}
    (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
