"""F-43m symmetry operations and Wyckoff-orbit helpers.

All helpers assume the cubic argyrodite space group F-43m. Adapt
:data:`F43M_OPS` and :func:`project_wyckoff` for other space groups / Wyckoff
sets.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray
from pymatgen.symmetry.groups import SpaceGroup

from .geometry import pbc_mic


F43M_OPS = SpaceGroup('F-43m').symmetry_ops


def super_to_prim(super_f: NDArray[np.float64],
                  supercell: tuple[int, int, int]) -> NDArray[np.float64]:
    """Collapse a supercell fractional coordinate into the primitive cell."""
    return (np.asarray(super_f, float) * np.asarray(supercell)) % 1.0


def prim_to_super(prim_orbit: NDArray[np.float64],
                  supercell: tuple[int, int, int]) -> NDArray[np.float64]:
    """Expand a primitive-cell orbit across the supercell translations."""
    sc = np.asarray(supercell)
    return np.array([
        (np.asarray(p) + np.array([i, j, k])) / sc
        for i, j, k in product(*(range(n) for n in supercell))
        for p in prim_orbit
    ])


def expand_orbit(canonical: NDArray[np.float64],
                 symops=F43M_OPS,
                 tol: float = 1e-4) -> NDArray[np.float64]:
    """Apply every symmetry op to ``canonical`` and return unique images under PBC."""
    orbit: list[NDArray[np.float64]] = []
    for op in symops:
        pos = op.operate(canonical) % 1.0
        if not any(np.linalg.norm(pbc_mic(pos - o)) < tol for o in orbit):
            orbit.append(pos)
    return np.array(orbit)


def fold_to_canonical(prim_f: NDArray[np.float64],
                      canonical: NDArray[np.float64],
                      symops=F43M_OPS) -> NDArray[np.float64]:
    """Apply every symmetry op to ``prim_f`` and return the image nearest to ``canonical``."""
    best_d, best_f = np.inf, prim_f.copy()
    for op in symops:
        folded = op.operate(prim_f) % 1.0
        d = np.linalg.norm(pbc_mic(folded - canonical))
        if d < best_d:
            best_d, best_f = d, folded
    return best_f


def circular_mean_frac(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Circular mean of fractional coordinates; safe across periodic boundaries."""
    cs = np.cos(2 * np.pi * positions).mean(axis=0)
    sn = np.sin(2 * np.pi * positions).mean(axis=0)
    return (np.arctan2(sn, cs) / (2 * np.pi)) % 1.0


def project_wyckoff(vec: NDArray[np.float64], label: str) -> NDArray[np.float64]:
    """Project onto the free-parameter surface of the given Wyckoff position.

    ``16e`` has the form ``(x, x, x)``; ``48h`` and ``48h'`` (here called
    ``48h2``) have ``(x, x, z)``. Unknown labels are returned unchanged.
    """
    v = np.asarray(vec, float)
    if label == '16e':
        return np.full(3, v.mean())
    if label in ('48h', '48h2'):
        xy = (v[0] + v[1]) / 2
        return np.array([xy, xy, v[2]])
    return v.copy()
