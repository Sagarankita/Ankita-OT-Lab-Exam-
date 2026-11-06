"""Slack QUBO and Ising coefficients with exhaustive encoding checks."""
import numpy as np

from ..data import TIMES


def qubo(predictions):
    """Upper-triangular polynomial; little-endian bit j is variable z_j."""
    A = B = float(1 + sum(abs(predictions)))
    c = np.r_[np.ones(8), np.zeros(4)]
    w = np.r_[TIMES, [1,2,4,8]]
    v = np.r_[predictions, np.zeros(4)]
    C = 9*A + 100*B
    h = -v + A*(c*c-6*c) + B*(w*w-20*w)
    J = np.triu(2*A*np.outer(c,c)+2*B*np.outer(w,w), 1)
    bits = ((np.arange(4096)[:,None] >> np.arange(12)) & 1).astype(float)
    energies = -(bits @ v) + A*(bits @ c-3)**2 + B*(bits @ w-10)**2
    expanded = C + bits @ h + np.einsum('bi,ij,bj->b', bits, J, bits)
    assert np.allclose(energies, expanded, atol=1e-8)
    # z=(1-Z)/2 gives a diagonal circuit with Rz and RZZ cost gates.
    ising_h = -h/2 - (J.sum(axis=0)+J.sum(axis=1))/4
    ising_J = J/4
    ising_C = C + sum(h)/2 + J.sum()/4
    spins = 1-2*bits
    gate_energy = ising_C + spins @ ising_h + np.einsum('bi,ij,bj->b', spins, ising_J, spins)
    assert np.allclose(energies, gate_energy, atol=1e-8)
    metadata = dict(variable_order=[f'x{i+1}' for i in range(8)]+[f'b{i}' for i in range(4)],
                    bit_order='z_j = (integer >> j) & 1; displayed strings are z0..z11',
                    A=A, B=B, constant=C, linear=h.tolist(), upper_quadratic=J.tolist(),
                    convention='C + sum(h[j]*z[j]) + sum(j<k,J[j,k]*z[j]*z[k])',
                    ising_h=ising_h.tolist(), ising_J=ising_J.tolist(),
                    ising_constant=float(ising_C), energy_scale=A,
                    cost_gates='Rz(2*gamma*ising_h/A), RZZ(2*gamma*ising_J/A); global phase omitted')
    return energies, bits, metadata
