"""Low-level geometry: PBC primitives, framework averaging, rigid shift, supercell replication."""
from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure


# ---------------------------------------------------------------------------
# Periodic boundary conditions
# ---------------------------------------------------------------------------

def pbc_mic(diff: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimum-image convention: wrap fractional differences into (-0.5, 0.5]."""
    return diff - np.round(diff)


def pbc_mean_frac(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Time-average fractional positions with PBC unwrapping against frame 0.

    Parameters
    ----------
    positions
        Array of shape ``(n_frames, n_atoms, 3)`` in fractional coordinates.

    Returns
    -------
    ndarray
        Shape ``(n_atoms, 3)``, mean fractional position per atom wrapped into
        ``[0, 1)``. An empty array is returned if ``n_atoms == 0``.
    """
    if positions.shape[1] == 0:
        return np.empty((0, 3))
    ref = positions[0]
    unwrapped = ref + pbc_mic(positions - ref)
    return unwrapped.mean(axis=0) % 1.0


# ---------------------------------------------------------------------------
# Rigid-shift recovery
# ---------------------------------------------------------------------------

def recover_rigid_shift(ideal_frac: NDArray[np.float64],
                        actual_frac: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate the rigid fractional shift that best aligns two point sets.

    For each atom in ``actual_frac`` the nearest atom in ``ideal_frac`` is
    found under PBC, giving a per-atom MIC displacement. The *circular* mean
    of those displacements is returned, so that shifts straddling the periodic
    boundary are handled correctly.

    Returns an array of shape ``(3,)`` with values in ``[0, 1)``.
    """
    if len(ideal_frac) == 0 or len(actual_frac) == 0:
        return np.zeros(3)
    diffs = pbc_mic(actual_frac[:, None, :] - ideal_frac[None, :, :])
    nearest = np.argmin((diffs ** 2).sum(axis=2), axis=1)
    shifts = diffs[np.arange(len(actual_frac)), nearest]
    cs = np.cos(2 * np.pi * shifts).mean(axis=0)
    sn = np.sin(2 * np.pi * shifts).mean(axis=0)
    return (np.arctan2(sn, cs) / (2 * np.pi)) % 1.0


def min_pbc_distances_A(source_frac: NDArray[np.float64],
                        target_frac: NDArray[np.float64],
                        shift_frac: NDArray[np.float64],
                        lattice: Lattice) -> NDArray[np.float64]:
    """Smallest PBC cartesian distance (Å) from each source atom to any target
    atom after subtracting ``shift_frac`` from the source.
    """
    diffs = pbc_mic(source_frac[:, None, :] - shift_frac - target_frac[None, :, :])
    cart = diffs @ np.asarray(lattice.matrix)
    return np.linalg.norm(cart, axis=-1).min(axis=1)


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------

def replicate_preserving_labels(struct: Structure,
                                supercell: tuple[int, int, int]) -> Structure:
    """Replicate ``struct`` across ``supercell = (nx, ny, nz)`` preserving site labels.

    pymatgen's ``Structure.make_supercell`` does not carry the per-site
    ``label`` field through replication, so this helper builds the supercell
    explicitly.
    """
    sc = np.asarray(supercell)
    lat = struct.lattice
    super_lat = Lattice.from_parameters(
        lat.a * sc[0], lat.b * sc[1], lat.c * sc[2],
        lat.alpha, lat.beta, lat.gamma,
    )
    species, coords, labels = [], [], []
    for ix, iy, iz in product(*(range(n) for n in supercell)):
        offset = np.array([ix, iy, iz])
        for site in struct.sites:
            species.append(site.specie)
            coords.append((site.frac_coords + offset) / sc)
            labels.append(site.label)
    return Structure(super_lat, species, coords, labels=labels,
                     coords_are_cartesian=False)


def ideal_fracs_by_label(structure: Structure, label: str) -> NDArray[np.float64]:
    """Fractional coordinates of every site in ``structure`` whose label matches."""
    return np.array([s.frac_coords for s in structure.sites if s.label == label])
