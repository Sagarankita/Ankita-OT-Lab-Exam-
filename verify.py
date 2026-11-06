#!/usr/bin/env python3
"""Independent checks for the custom quantum simulator and saved submission.

Run after solution.py: python3 verify.py. Does not rerun quantum optimization.
"""
import csv
import json
import re
import unittest
from collections import Counter
from itertools import combinations

import numpy as np
import solution as s


def read_csv(name):
    with (s.OUT/name).open(newline='') as f:
        return list(csv.DictReader(f))


class SubmissionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=json.loads((s.OUT/'results.json').read_text())
        cls.metadata=json.loads((s.OUT/'qubo.json').read_text())

    def test_candidate_inputs_and_regression(self):
        expected=[[14,35,.8,6],[9,50,.3,12],[18,20,.6,25],[6,55,.9,8],
                  [16,45,.5,10],[11,30,.7,18],[4,40,.2,30],[12,25,.4,15]]
        np.testing.assert_array_equal(s.CANDIDATES,expected)
        np.testing.assert_array_equal(s.TIMES,[4,2,3,5,4,3,3,3])
        # Independently fit the exact generated data and verify saved predictions.
        X,y=s.training_data()
        self.assertEqual(X.shape,(120,4)); self.assertEqual(y.shape,(120,))
        coefficients=np.linalg.lstsq(np.c_[np.ones(120),X],y,rcond=None)[0]
        np.testing.assert_allclose(coefficients,self.result['coefficients'],atol=1e-12)
        predictions=np.c_[np.ones(8),s.CANDIDATES]@coefficients
        np.testing.assert_allclose(predictions,self.result['predictions'],atol=1e-12)
        saved=read_csv('candidate_predictions.csv')
        np.testing.assert_allclose([float(r['predicted_damage']) for r in saved],predictions)

    def test_all_selections_and_optimum(self):
        rows=read_csv('all_selections.csv')
        self.assertEqual(len(rows),56)
        self.assertEqual(len({r['locations'] for r in rows}),56)
        predicted=self.result['predictions']
        independently_feasible=[]
        for triple,row in zip(combinations(range(8),3),rows):
            expected_names=','.join(f'L{i+1}' for i in triple)
            self.assertEqual(row['locations'],expected_names)
            score=sum(predicted[i] for i in triple)
            hours=sum(s.TIMES[i] for i in triple)
            self.assertAlmostEqual(float(row['predicted_damage']),score,places=10)
            self.assertEqual(int(row['hours']),hours)
            self.assertEqual(row['feasible']=='True',hours<=10)
            if hours<=10: independently_feasible.append((score,triple))
        self.assertEqual(len(independently_feasible),35)
        score,triple=max(independently_feasible)
        self.assertEqual(list(triple),[0,1,4])
        self.assertAlmostEqual(score,self.result['optimum']['damage'],places=10)
        self.assertEqual(self.result['comparison'][0]['hours'],13)

    def test_quantum_gate_primitives(self):
        state=s.simulate_gates([('h',(3,),0),('h',(3,),0)])
        self.assertAlmostEqual(abs(state[0]),1)
        state=s.simulate_gates([('rx',(3,),np.pi)])
        self.assertAlmostEqual(abs(state[8]),1)
        bell=s.simulate_gates([('h',(0,),0),('cx',(0,1),0)])
        np.testing.assert_allclose(abs(bell[[0,3]])**2,[.5,.5],atol=1e-12)
        self.assertAlmostEqual(float(sum(abs(bell)**2)),1)
        phased=s.simulate_gates([('rz',(2,),.7)])
        self.assertAlmostEqual(abs(phased[0]-np.exp(-.35j)),0)

    def test_exported_elementary_circuit(self):
        # Parse the exported file independently of the code generating the gates.
        gates=[]
        for line in (s.OUT/'circuit.qasm').read_text().splitlines():
            if line.startswith(('//','OPENQASM','include','qreg','creg','measure')): continue
            single=re.fullmatch(r'(h|rx|rz)(?:\(([^)]+)\))? q\[(\d+)\];',line)
            pair=re.fullmatch(r'cx q\[(\d+)\],q\[(\d+)\];',line)
            if single:
                gates.append((single[1],(int(single[3]),),float(single[2] or 0)))
            elif pair:
                gates.append(('cx',(int(pair[1]),int(pair[2])),0))
            else: self.fail(f'Unexpected QASM line: {line}')
        self.assertEqual(dict(Counter(g[0] for g in gates)),self.metadata['gate_counts'])
        energies,bits,metadata=s.qubo(np.array(self.result['predictions']))
        cost=(energies-metadata['constant'])/metadata['A']
        fused=s.statevector(self.metadata['chosen']['parameters'],cost)
        explicit=s.simulate_gates(gates)
        self.assertAlmostEqual(float(abs(np.vdot(fused,explicit))**2),1,places=10)
        ground=np.flatnonzero(np.isclose(energies,energies.min(),rtol=0,atol=1e-8))
        self.assertEqual(len(ground),1)
        self.assertEqual(np.flatnonzero(bits[ground[0],:8]).tolist(),[0,1,4])

    def test_shots_and_marginalization(self):
        samples=read_csv('quantum_samples.csv'); selections=read_csv('quantum_selections.csv')
        self.assertEqual(sum(int(r['count']) for r in samples),4096)
        self.assertEqual(sum(int(r['count']) for r in selections),4096)
        self.assertEqual(len(selections),256)
        self.assertAlmostEqual(sum(float(r['probability']) for r in selections),1,places=10)
        feasible_count=0; optimal_count=0
        for row in samples:
            bits=np.array(list(map(int,row['z0_to_z11'])))
            self.assertEqual(len(bits),12)
            hours=int(bits[:8]@s.TIMES)
            feasible=bits[:8].sum()==3 and hours<=10
            self.assertEqual(row['original_feasible']=='True',feasible)
            encoded=bits[:8].sum()==3 and hours+int(bits[8:]@[1,2,4,8])==10
            self.assertEqual(row['encoded_feasible']=='True',encoded)
            if feasible: feasible_count+=int(row['count'])
            if np.flatnonzero(bits[:8]).tolist()==[0,1,4]: optimal_count+=int(row['count'])
        self.assertAlmostEqual(feasible_count/4096,self.result['quantum']['feasible_fraction'])
        self.assertEqual(optimal_count,self.result['quantum']['optimum_count'])
        self.assertTrue((s.OUT/'results.pdf').read_bytes().startswith(b'%PDF-'))


if __name__=='__main__':
    unittest.main(verbosity=2)
