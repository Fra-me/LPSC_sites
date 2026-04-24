"""LPSC site extraction from AIMD density volumes.

Anchor-based Li site extraction for cubic argyrodite Li6PS5Cl (F-43m):
candidate positions are generated from the reference CIF's Wyckoff orbits,
rigidly aligned to the AIMD frame, and refined to the local density maximum
with an iterative empirical-canonical learning loop.

Public API
----------
Records
    :class:`Candidate`, :class:`CageStatus`
Geometry
    :func:`pbc_mic`, :func:`pbc_mean_frac`, :func:`recover_rigid_shift`,
    :func:`min_pbc_distances_A`, :func:`replicate_preserving_labels`,
    :func:`ideal_fracs_by_label`
Density
    :class:`DensityVolume`, :func:`refine_candidates`
Symmetry (F-43m)
    :data:`F43M_OPS`, :func:`super_to_prim`, :func:`prim_to_super`,
    :func:`expand_orbit`, :func:`fold_to_canonical`,
    :func:`circular_mean_frac`, :func:`project_wyckoff`
Pipeline
    :func:`nearest_species_at`, :func:`classify_cages`,
    :func:`nearest_cage_idx`, :func:`generate_candidates`,
    :func:`reset_candidates_to_cif`, :func:`learn_empirical_canonicals`,
    :func:`find_duplicate_conflicts`, :func:`build_labeled_structure`,
    :func:`export_metadata`
"""
from __future__ import annotations

from .density import DensityVolume, refine_candidates
from .geometry import (
    ideal_fracs_by_label,
    min_pbc_distances_A,
    pbc_mean_frac,
    pbc_mic,
    recover_rigid_shift,
    replicate_preserving_labels,
)
from .pipeline import (
    build_labeled_structure,
    classify_cages,
    export_metadata,
    find_duplicate_conflicts,
    generate_candidates,
    learn_empirical_canonicals,
    nearest_cage_idx,
    nearest_species_at,
    reset_candidates_to_cif,
)
from .records import CageStatus, Candidate
from .symmetry import (
    F43M_OPS,
    circular_mean_frac,
    expand_orbit,
    fold_to_canonical,
    prim_to_super,
    project_wyckoff,
    super_to_prim,
)

__version__ = '0.1.0'

__all__ = [
    # records
    'CageStatus', 'Candidate',
    # geometry
    'pbc_mic', 'pbc_mean_frac', 'recover_rigid_shift', 'min_pbc_distances_A',
    'replicate_preserving_labels', 'ideal_fracs_by_label',
    # density
    'DensityVolume', 'refine_candidates',
    # symmetry
    'F43M_OPS', 'super_to_prim', 'prim_to_super',
    'expand_orbit', 'fold_to_canonical',
    'circular_mean_frac', 'project_wyckoff',
    # pipeline
    'nearest_species_at', 'classify_cages',
    'nearest_cage_idx', 'generate_candidates',
    'reset_candidates_to_cif', 'learn_empirical_canonicals',
    'find_duplicate_conflicts', 'build_labeled_structure',
    'export_metadata',
]
