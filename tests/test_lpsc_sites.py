"""Tests for lpsc_sites.

Focused on the mathematical invariants that the pipeline relies on: PBC
handling, circular statistics, Wyckoff projection, symmetry orbits, and
density-volume queries. Not a coverage target — every test here encodes a
property that, if broken, would silently corrupt extracted site positions.
"""
from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice

from lpsc_sites import (
    Candidate,
    DensityVolume,
    circular_mean_frac,
    expand_orbit,
    find_duplicate_conflicts,
    fold_to_canonical,
    pbc_mean_frac,
    pbc_mic,
    project_wyckoff,
    recover_rigid_shift,
)
from lpsc_sites.symmetry import F43M_OPS


# ---------------------------------------------------------------------------
# PBC primitives
# ---------------------------------------------------------------------------

def test_pbc_mic_wraps_to_half_open_interval():
    # Inputs in (-1, 1), output must be in (-0.5, 0.5]
    diffs = np.array([[0.6, -0.6, 0.4],
                      [0.501, -0.501, 0.5],
                      [0.0, 0.999999, -0.999999]])
    out = pbc_mic(diffs)
    assert np.all(out > -0.5 - 1e-12)
    assert np.all(out <= 0.5 + 1e-12)
    np.testing.assert_allclose(out[0], [-0.4, 0.4, 0.4])


def test_pbc_mean_frac_empty_returns_empty():
    out = pbc_mean_frac(np.empty((10, 0, 3)))
    assert out.shape == (0, 3)


def test_pbc_mean_frac_straddles_boundary():
    # Atom oscillating around x=0 (i.e. across the 0/1 boundary):
    # frac coords of 0.99, 0.01, 0.99, 0.01 should mean to ≈ 0 (not 0.5).
    positions = np.array([
        [[0.99, 0.5, 0.5]],
        [[0.01, 0.5, 0.5]],
        [[0.99, 0.5, 0.5]],
        [[0.01, 0.5, 0.5]],
    ])
    mean = pbc_mean_frac(positions)
    assert mean.shape == (1, 3)
    # The unwrapped mean should sit next to the boundary, not at the middle
    dist_to_boundary = min(mean[0, 0], 1.0 - mean[0, 0])
    assert dist_to_boundary < 0.05


# ---------------------------------------------------------------------------
# Rigid-shift recovery
# ---------------------------------------------------------------------------

def test_recover_rigid_shift_recovers_known_shift():
    # Four well-separated atoms in an FCC arrangement (mimics P anchor in the
    # LPSC primitive cell). With dense random atoms a large shift would confuse
    # the nearest-neighbour lookup — realistic P-anchor shifts are small, so
    # this geometry matches the actual use case.
    ideal = np.array([
        [0.00, 0.00, 0.00],
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
    ])
    true_shift = np.array([0.05, 0.08, 0.03])
    actual = (ideal + true_shift) % 1.0
    recovered = recover_rigid_shift(ideal, actual)
    err = np.linalg.norm(pbc_mic(recovered - true_shift))
    assert err < 1e-10


def test_recover_rigid_shift_handles_boundary_straddling():
    ideal = np.array([
        [0.00, 0.00, 0.00],
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
    ])
    # Shift with the last component just under the boundary — under MIC it's
    # effectively -0.001, which must be handled by the circular mean.
    true_shift = np.array([0.0, 0.001, 0.999])
    actual = (ideal + true_shift) % 1.0
    recovered = recover_rigid_shift(ideal, actual)
    err = np.linalg.norm(pbc_mic(recovered - true_shift))
    assert err < 1e-10


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def test_circular_mean_frac_across_boundary():
    # Points clustered either side of the 0/1 boundary should mean to ~0
    points = np.array([[0.01, 0.5, 0.5],
                       [0.99, 0.5, 0.5],
                       [0.99, 0.5, 0.5]])
    m = circular_mean_frac(points)
    dist = min(m[0], 1.0 - m[0])
    assert dist < 0.02


# ---------------------------------------------------------------------------
# Wyckoff projection
# ---------------------------------------------------------------------------

def test_project_wyckoff_16e_is_diagonal():
    v = np.array([0.1, 0.25, 0.13])
    p = project_wyckoff(v, '16e')
    assert np.allclose(p, [v.mean()] * 3)


def test_project_wyckoff_48h_is_xxz():
    v = np.array([0.3, 0.5, 0.7])
    p = project_wyckoff(v, '48h')
    assert p[0] == p[1] == pytest.approx(0.4)
    assert p[2] == pytest.approx(0.7)

    # 48h2 gets the same treatment
    p2 = project_wyckoff(v, '48h2')
    np.testing.assert_allclose(p, p2)


def test_project_wyckoff_unknown_label_is_identity():
    v = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(project_wyckoff(v, 'something'), v)


# ---------------------------------------------------------------------------
# Symmetry orbits
# ---------------------------------------------------------------------------

def test_expand_orbit_images_are_unique_under_pbc():
    # In F-43m primitive cell, Wyckoff 16e (multiplicity 16 in conventional, 4 in primitive)
    canonical = np.array([0.25, 0.25, 0.25])
    orbit = expand_orbit(canonical, symops=F43M_OPS)
    assert orbit.shape[1] == 3
    # Orbit members must all be unique under PBC
    for i in range(len(orbit)):
        for j in range(i + 1, len(orbit)):
            d = np.linalg.norm(pbc_mic(orbit[i] - orbit[j]))
            assert d > 1e-3, f"duplicate orbit members at indices {i}, {j}"


def test_fold_to_canonical_recovers_canonical_image():
    canonical = np.array([0.125, 0.125, 0.375])  # 48h-like free parameters
    # For every symmetry op, applying it to the canonical and then folding
    # back should return an image coinciding with the canonical under PBC.
    for op in F43M_OPS:
        shifted = op.operate(canonical) % 1.0
        folded = fold_to_canonical(shifted, canonical, symops=F43M_OPS)
        err = np.linalg.norm(pbc_mic(folded - canonical))
        assert err < 1e-6, f"fold failed after op {op.as_xyz_str()}: err={err}"


# ---------------------------------------------------------------------------
# DensityVolume
# ---------------------------------------------------------------------------

def _make_toy_volume(n: int = 40, peak_frac=(0.3, 0.5, 0.7), sigma: float = 0.05):
    """Gaussian peak on an n^3 cubic grid with a = 10 Å."""
    lattice = Lattice.cubic(10.0)
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    dx = X - peak_frac[0]
    dy = Y - peak_frac[1]
    dz = Z - peak_frac[2]
    density = np.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * sigma ** 2))
    return DensityVolume(density, lattice)


def test_density_volume_local_max_finds_peak():
    peak = (0.3, 0.5, 0.7)
    vol = _make_toy_volume(peak_frac=peak)
    # Start away from the peak within the ROI
    start = np.array([peak[0] + 0.05, peak[1] - 0.03, peak[2] + 0.02])
    refined, value, disp = vol.local_max(start, radius_A=1.0)
    # Refined voxel centre must be within one voxel-diagonal of the true peak
    voxel_diag = np.linalg.norm(vol.voxel_size_A)
    assert np.linalg.norm(pbc_mic(refined - np.array(peak)) @ vol._matrix) < 1.5 * voxel_diag
    assert value > 0.9  # Gaussian max is 1


def test_density_volume_integrate_scales_with_volume():
    vol = _make_toy_volume()
    # Uniform-looking small sphere: integrate at the peak with two radii,
    # larger should be >= smaller (and strictly larger unless on voxel-lattice edge).
    peak = np.array([0.3, 0.5, 0.7])
    total_small, n_small = vol.integrate(peak, radius_A=0.5)
    total_large, n_large = vol.integrate(peak, radius_A=1.0)
    assert n_large > n_small
    assert total_large > total_small


def test_density_volume_plot_3d_raises_without_source():
    vol = _make_toy_volume()
    with pytest.raises(RuntimeError, match='plot_3d'):
        vol.plot_3d()


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def test_find_duplicate_conflicts_flags_close_pairs():
    lattice = Lattice.cubic(10.0)
    # Three candidates: 0 and 1 close (0.1 Å apart), 2 far
    cands = [
        Candidate(wyckoff='48h', full_label='48h', cage_type='S_4d',
                  cage_idx=0, cage_status='normal',
                  ideal_frac=np.zeros(3), candidate_frac=np.zeros(3),
                  refined_frac=np.array([0.10, 0.10, 0.10])),
        Candidate(wyckoff='48h', full_label='48h', cage_type='S_4d',
                  cage_idx=0, cage_status='normal',
                  ideal_frac=np.zeros(3), candidate_frac=np.zeros(3),
                  refined_frac=np.array([0.101, 0.10, 0.10])),  # 0.01 * 10 Å = 0.1 Å away
        Candidate(wyckoff='16e', full_label='16e', cage_type='S_4d',
                  cage_idx=1, cage_status='normal',
                  ideal_frac=np.zeros(3), candidate_frac=np.zeros(3),
                  refined_frac=np.array([0.50, 0.50, 0.50])),
    ]
    pairs = find_duplicate_conflicts(cands, lattice, threshold_A=0.3)
    assert pairs == [(0, 1)]
    assert cands[0].conflicts_with == [1]
    assert cands[1].conflicts_with == [0]
    assert cands[2].conflicts_with == []
