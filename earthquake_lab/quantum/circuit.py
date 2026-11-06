"""Fused QAOA states, elementary-gate simulation, and OpenQASM export."""
import numpy as np


def statevector(params, cost):
    """H^12; repeat diagonal cost circuit and tensor-product Rx(2 beta).

    Cost gates commute: applying exp(-i gamma cost) is exactly their product.
    No energy minimizer or feasibility postselection is used to prepare the state.
    Indexing is little-endian: qubit 0 changes fastest.
    """
    state = np.full(4096, 1/64, dtype=complex)  # H on each |0> qubit
    for gamma, beta in np.asarray(params).reshape(-1,2):
        state *= np.exp(-1j*gamma*cost)
        cs, sn = np.cos(beta), -1j*np.sin(beta)
        for q in range(12):
            blocks = state.reshape(-1,2,1 << q)
            zero, one = blocks[:,0,:].copy(), blocks[:,1,:].copy()
            blocks[:,0,:] = cs*zero + sn*one
            blocks[:,1,:] = sn*zero + cs*one
    return state


def gate_circuit(params, metadata):
    """Compile cost layers to H/Rz/CX/Rx, without a problem-specific oracle.

    exp(-i*gamma*J*Zi*Zj) = CX(i,j) Rz(2*gamma*J,j) CX(i,j).
    This independent gate path is used for the final state and measurements.
    """
    gates = [('h',(q,),0.0) for q in range(12)]
    scale = metadata['energy_scale']
    for gamma,beta in np.asarray(params).reshape(-1,2):
        for q,h in enumerate(metadata['ising_h']):
            gates.append(('rz',(q,),float(2*gamma*h/scale)))
        for i in range(12):
            for j in range(i+1,12):
                angle = float(2*gamma*metadata['ising_J'][i][j]/scale)
                if abs(angle) > 1e-14:
                    gates.extend([('cx',(i,j),0.0),('rz',(j,),angle),('cx',(i,j),0.0)])
        gates.extend(('rx',(q,),float(2*beta)) for q in range(12))
    return gates


def simulate_gates(gates):
    """Elementary-gate simulator independent of the fused cost-phase kernel."""
    state = np.zeros(4096,dtype=complex); state[0] = 1
    indices = np.arange(4096)
    for gate,wires,angle in gates:
        q = wires[0]
        if gate == 'cx':
            target = wires[1]
            zero = indices[((indices >> q)&1).astype(bool) & (((indices >> target)&1)==0)]
            one = zero | (1 << target)
            state[zero],state[one] = state[one].copy(),state[zero].copy()
            continue
        blocks = state.reshape(-1,2,1 << q)
        zero,one = blocks[:,0,:].copy(),blocks[:,1,:].copy()
        if gate == 'h':
            blocks[:,0,:],blocks[:,1,:] = (zero+one)/np.sqrt(2),(zero-one)/np.sqrt(2)
        elif gate == 'rz':
            blocks[:,0,:],blocks[:,1,:] = np.exp(-.5j*angle)*zero,np.exp(.5j*angle)*one
        elif gate == 'rx':
            cs,sn = np.cos(angle/2),-1j*np.sin(angle/2)
            blocks[:,0,:],blocks[:,1,:] = cs*zero+sn*one,sn*zero+cs*one
        else:
            raise ValueError(f'Unsupported gate: {gate}')
    return state


def qasm_text(gates):
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[12];', 'creg c[12];',
             '// q[0..7]: L1..L8; q[8..11]: slack weights 1,2,4,8.',
             '// All bitstrings in output.txt are displayed q[0] first.']
    for gate,wires,angle in gates:
        if gate=='cx': lines.append(f'cx q[{wires[0]}],q[{wires[1]}];')
        elif gate=='h': lines.append(f'h q[{wires[0]}];')
        else: lines.append(f'{gate}({angle:.17g}) q[{wires[0]}];')
    lines.append('measure q -> c;')
    return '\n'.join(lines)+'\n'
