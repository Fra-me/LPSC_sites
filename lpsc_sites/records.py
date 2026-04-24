"""Typed record classes used throughout the site-extraction pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class CageStatus:
    """Classification of one 4a / 4d cage centre in the AIMD framework.

    Attributes
    ----------
    idx
        Index of the cage in the ideal cage-centre array.
    target
        Expected fractional position of the cage centre (ideal + rigid shift).
    species
        Symbol of the framework species found closest to ``target``.
    distance_A
        Cartesian distance (Å) from ``target`` to that species.
    status
        ``'normal'`` if ``species`` belongs to the expected set (Cl at 4a,
        S/O at 4d), otherwise ``'swapped'``.
    """

    idx: int
    target: NDArray[np.float64]
    species: str | None
    distance_A: float
    status: str


@dataclass(slots=True)
class Candidate:
    """One Li candidate site with its Wyckoff label and refinement state.

    The candidate is initialised with just the starting position and Wyckoff
    label; the refinement-related fields are populated by
    :func:`lpsc_sites.density.refine_candidates` and the conflict detection
    step.
    """

    wyckoff: str                         # '48h', '48h2', '16e'
    full_label: str                      # mirrors wyckoff; exposed for downstream use
    cage_type: str                       # framework site that labels it (e.g. 'S_4d')
    cage_idx: int                        # index into the corresponding ideal cage array
    cage_status: str                     # 'normal' or 'swapped' (from associated cage)
    ideal_frac: NDArray[np.float64]      # current "ideal" start (may be updated by learning)
    candidate_frac: NDArray[np.float64]  # ideal_frac shifted into the AIMD frame

    refined_frac: NDArray[np.float64] | None = None
    refined_density: float | None = None
    refinement_disp_A: float | None = None
    integrated_density: float | None = None
    integrated_voxels: int | None = None
    conflicts_with: list[int] = field(default_factory=list)
