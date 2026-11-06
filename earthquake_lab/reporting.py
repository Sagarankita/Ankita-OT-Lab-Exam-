"""Readable console output and the one-page Markdown/PDF results sheet."""
import json
import platform

import numpy as np

from .classical import feasible, names
from .config import OUT, REFERENCES, SHOTS
from .data import CANDIDATES, TIMES


def print_predictions(result):
    predictions = result['predictions']
    metrics = result['metrics']
    equation = result['equation']
    print('EARTHQUAKE INSPECTION - REPRODUCIBLE RUN')
    print(f'Python {platform.python_version()}, NumPy {np.__version__}; exact data seed=63; split seed=7')
    print('Validation: 96 train / 24 test:',json.dumps(metrics))
    print('All-record regression:',equation)
    print('Location | predicted damage | hours')
    for i,(pred,hours) in enumerate(zip(predictions,TIMES),1): print(f'L{i}       {pred:.6f}            {hours}')


def print_results(result):
    q = result['quantum']
    optimum = result['optimum']
    comparison = result['comparison']
    feasible_count = result['feasible_count']
    ml_seconds = result['ml_seconds']
    print('\nMETHOD                   LOCATIONS      DAMAGE      HOURS FEASIBLE RUNTIME(s)')
    for row in comparison: print(f"{row['method']:<24} {names(row):<13} {row['damage']:10.6f} {row['hours']:5} {str(feasible(row)):8} {row['seconds']:.6f}")
    print(f'Common ML/data preparation: {ml_seconds:.6f} s (excluded from solver runtimes)')
    print('Quantum encoding + verification + optimization + sampling included; PDF/report I/O excluded.')
    print('\nQAOA attempts:',json.dumps(q['metadata']['attempts'],indent=2))
    print('Cost scaling A=B=',q['metadata']['A'],'; 12 qubits; shots=4096; sample seed=2026')
    print('Initialization seed=29; total objective evaluations=',q['metadata']['total_evaluations'])
    print('Elementary circuit gate counts:',q['metadata']['gate_counts'],'; fidelity=',q['metadata']['circuit_fidelity'])
    print('Most frequent full bitstring (z0..z11):', ''.join(str(int(x)) for x in q['bits'][q['modal_state']['state']]))
    print('Modal full state:',names(q['modal_state']),'; hours=',q['modal_state']['hours'],'; original feasible=',feasible(q['modal_state']),'; encoded feasible=',q['modal_state']['encoded_feasible'])
    print('Modal measured location selection (slack marginalized):',names(q['modal_selection']),'; feasible=',feasible(q['modal_selection']))
    print(f"Original feasible shots: {q['feasible_fraction']:.6%}; encoded feasible shots: {q['encoded_feasible_fraction']:.6%}")
    print(f"Exact feasible probability: {q['feasible_probability']:.6%}; optimum selection probability: {q['optimum_probability']:.6%}; measured optimum fraction: {q['optimum_fraction']:.6%} ({q['optimum_count']}/{SHOTS})")
    print(f'Uniform 12-qubit reference: original feasible probability={feasible_count/256:.6%}; unique optimum selection probability={1/256:.6%} (all slack values marginalized)')
    if q['best']:
        gap=optimum['damage']-q['best']['damage']
        print(f'Best feasible sample damage gap={gap:.9f} ({100*gap/optimum["damage"]:.6f}%)')
    else: print('No feasible QAOA sample; quantum result unavailable.')
    print('CHECKS PASSED:',q['metadata']['checks'],'; parallel equals sequential.')
    print('No quantum speedup claimed. Single-run timings vary; process overhead dominates 56 triples.')
    print('Two limitations: synthetic data/generalization; shallow heuristic QAOA and exponential statevector scaling.')
    print('References:')
    for title,url in REFERENCES: print(title,url)


def create_reports(result):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    p = result['predictions']; m = result['metrics']; q = result['quantum']; opt = result['optimum']
    rows = result['comparison']
    eq = result['equation']
    qbest = q['best']
    match = qbest is not None and np.isclose(qbest['damage'],opt['damage'])
    gap = opt['damage']-qbest['damage'] if qbest else None
    interpretation = (f"The best feasible QAOA sample {'matches' if match else 'does not match'} exhaustive search. "
                      + (f"Damage gap: {gap:.6f} ({100*gap/opt['damage']:.4f}%). " if gap is not None else 'No feasible sample was obtained. ')
                      + f"Optimum selection: {q['optimum_count']}/{SHOTS} shots ({q['optimum_fraction']:.2%}); feasible shots: {q['feasible_fraction']:.2%}. "
                      + f"The modal full bitstring selects {names(q['modal_state'])}; the modal selection after combining slack outcomes is {names(q['modal_selection'])} (feasible: {feasible(q['modal_selection'])}). "
                      + 'Observing the optimum among 4096 shots does not establish concentration or quantum advantage.')
    formulation = (f"Binary x_i indicates inspection. Maximize sum(yhat_i*x_i), subject to sum(x_i)=3 and sum(t_i*x_i)<=10. "
                   f"There are C(8,3)=56 triples; {result['feasible_count']} meet the time limit.")
    ml = (f"OLS with intercept matches the linear synthetic generator (seed 63, N=120). "
          f"Independent 80/20 split (seed 7): MAE {m['mae']:.4f}, RMSE {m['rmse']:.4f}, R2 {m['r2']:.4f}. "
          f"Final fit uses all 120 records: {eq}.")
    quantum = (f"Let s=b0+2b1+4b2+8b3. Minimize E=-sum(yhat_i*x_i)+A(sum(x_i)-3)^2+B(sum(t_i*x_i)+s-10)^2. "
               f"A=B=1+sum(abs(yhat_i))={q['metadata']['A']:.4f}; a unit violation costs more than the reward range. "
               'Nonnegative slack prevents over-time solutions. The QUBO uses 12 qubits; all 4096 energies were checked against its expanded polynomial and Ising form. '
               f"QAOA: |+>^12, alternating exp(-i*gamma*E/A) and Rx(2*beta); depths 1,2,3,4,6, three starts each, {q['metadata']['total_evaluations']} optimizer evaluations total; chosen p={q['metadata']['chosen']['depth']}. "
               'Exact expectations train angles. The final elementary H/Rz/CX/Rx circuit is simulated independently (fidelity > 1-1e-10) and sampled for 4096 shots, seed 2026. Circuit: circuit.qasm; coefficients/settings: qubo.json.')
    parallel = (f"A spawn multiprocessing Pool with 2 workers evaluates each triple once; a sequential pass confirms the optimum. "
                'Runtimes below are single-run wall times: greedy sorting/evaluation; quantum encoding, validation, optimization and sampling; parallel pool startup/evaluation/reduction. '
                f"Common data/ML time is {result['ml_seconds']:.6f} s and is excluded. Process overhead dominates this tiny workload.")
    limitations = ('1. Synthetic linear data and a small held-out set do not establish real earthquake accuracy; prediction uncertainty may change the selected triple. '
                   '2. Shallow QAOA and a bounded local optimizer offer no convergence guarantee; classical statevector memory grows exponentially. No hardware noise is modeled.')
    md = ['# Earthquake inspection - results', '', '## Formulation', formulation, '', '## Connected solution', ml, '',
          '| Location | d (thousand/km2) | a (years) | l (0-1) | r (km) | Time (h) | Predicted damage |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for i,(features,pred,hours) in enumerate(zip(CANDIDATES,p,TIMES),1):
        md.append(f'| L{i} | {features[0]:g} | {features[1]:g} | {features[2]:g} | {features[3]:g} | {hours} | {pred:.4f} |')
    md += ['',quantum,'',parallel,'','## Measured comparison',
           '| Method | Locations | Damage | Hours | Time limit met? | Runtime (s) |', '|---|---|---:|---:|---|---:|']
    for row in rows:
        md.append(f"| {row['method']} | {names(row)} | {row['damage']:.6f} | {row['hours']} | {feasible(row)} | {row['seconds']:.6f} |")
    md += ['', '## Interpretation', interpretation, '', '## Two limitations', limitations, '', '## References']
    md += [f'{i}. [{title}]({url})' for i,(title,url) in enumerate(REFERENCES,1)]
    (OUT/'results.md').write_text('\n'.join(md)+'\n')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall',fontName='Helvetica',fontSize=9,leading=11.3,spaceAfter=5))
    styles.add(ParagraphStyle(name='SmallHead',fontName='Helvetica-Bold',fontSize=9.3,leading=11.5,spaceBefore=5,spaceAfter=4,textColor=colors.HexColor('#12384a')))
    styles.add(ParagraphStyle(name='RefSmall',fontName='Helvetica',fontSize=7,leading=9,spaceAfter=2))
    story=[]
    def para(text,style='BodySmall'):
        # Text contains ASCII <=; escape it for ReportLab's XML parser.
        from xml.sax.saxutils import escape
        story.append(Paragraph(escape(text),styles[style]))
    styles['Title'].fontSize=17
    styles['Title'].leading=21
    para('EARTHQUAKE INSPECTION - RESULTS','Title')
    para('Machine learning + quantum optimization + parallel verification','BodySmall')
    para('1  Formulation','SmallHead'); para(formulation)
    para('2  Working solution across all three domains','SmallHead'); para(ml)
    candidate_data=[['Location','d (1000/km2)','a (years)','l (0-1)','r (km)','Time (h)','Predicted damage']]
    for i,(features,pred,hours) in enumerate(zip(CANDIDATES,p,TIMES),1):
        candidate_data.append([f'L{i}',f'{features[0]:g}',f'{features[1]:g}',f'{features[2]:g}',f'{features[3]:g}',str(hours),f'{pred:.4f}'])
    candidates_table=Table(candidate_data,colWidths=[45,80,65,60,60,65,128])
    candidates_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),('ALIGN',(1,0),(-1,-1),'RIGHT'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3f6')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#bccbd3'))]))
    story.append(candidates_table)
    para(quantum); para(parallel)
    para('3  Measured comparison','SmallHead')
    data=[['Method','Selected locations','Damage','Hours','Time <=10?','Seconds']]
    for row in rows:
        data.append([row['method'],names(row),f"{row['damage']:.4f}",str(row['hours']),'Yes' if feasible(row) else 'No',f"{row['seconds']:.6f}"])
    table=Table(data,colWidths=[104,115,65,42,52,65],repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#12384a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf3f6'),colors.white]),('ALIGN',(2,0),(-1,-1),'RIGHT')]))
    story.append(table)
    para('4  Interpretation and limitations','SmallHead'); para(interpretation); para(limitations)
    para('References','SmallHead')
    for i,(title,url) in enumerate(REFERENCES,1):
        story.append(Paragraph(f'{i}. {title}: <link href="{url}" color="#145579">{url}</link>',styles['RefSmall']))
    doc=SimpleDocTemplate(str(OUT/'results.pdf'),pagesize=A4,rightMargin=26,leftMargin=26,topMargin=24,bottomMargin=24)
    doc.build(story)
    assert doc.page == 1, 'The results sheet must remain exactly one page.'
