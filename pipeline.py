"""High-level pipeline: cage classification, candidate generation, iterative
canonical learning, conflict detection, and metadata export.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice, Structure

from .density import DensityVolume
from .geometry import pbc_mic
from .records import CageStatus, Candidate
from .symmetry import (
    circular_mean_frac,
    expand_orbit,
    fold_to_canonical,
    prim_to_super,
    project_wyckoff,
    super_to_prim,
)


# ---------------------------------------------------------------------------
# Cage classification
# ---------------------------------------------------------------------------

def nearest_species_at(target_frac: NDArray[np.float64],
                       framework_by_species: dict[str, NDArray[np.float64]],
                       lattice: Lattice) -> tuple[str | None, float]:
    """Framework species closest to ``target_frac`` under PBC, plus distance (Å)."""
    best_sp, best_d = None, np.inf
    for sym, arr in framework_by_species.items():
        if len(arr) == 0:
            continue
        diffs = pbc_mic(arr - target_frac)
        dmin = np.linalg.norm(diffs @ np.asarray(lattice.matrix), axis=1).min()
        if dmin < best_d:
            best_d, best_sp = dmin, sym
    return best_sp, float(best_d)


def classify_cages(ideal_positions: NDArray[np.float64],
                   framework: dict[str, NDArray[np.float64]],
                   shift_frac: NDArray[np.float64],
                   lattice: Lattice,
                   normal_set: Iterable[str]) -> list[CageStatus]:
    """Classify each cage centre as ``'normal'`` or ``'swapped'``.

    For each ideal cage position, locate the closest framework atom in the
    AIMD frame (after applying ``shift_frac``) and compare its species to
    ``normal_set``.
    """
    normal = frozenset(normal_set)
    cages: list[CageStatus] = []
    for i, pos in enumerate(ideal_positions):
        target = (pos + shift_frac) % 1.0
        sp, d = nearest_species_at(target, framework, lattice)
        cages.append(CageStatus(
            idx=i,
            target=target,
            species=sp,
            distance_A=d,
            status='normal' if sp in normal else 'swapped',
        ))
    return cages


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def nearest_cage_idx(li_frac: NDArray[np.float64],
                     cage_fracs: NDArray[np.float64]) -> int:
    """Index of the cage centre closest to ``li_frac`` under PBC."""
    diffs = pbc_mic(cage_fracs - li_frac)
    return int(np.argmin(np.linalg.norm(diffs, axis=1)))


def generate_candidates(ideal_by_label: dict[str, NDArray[np.float64]],
                        li_labels: Iterable[str],
                        cage_status: list[CageStatus],
                        cage_fracs: NDArray[np.float64],
                        shift_frac: NDArray[np.float64],
                        cage_type: str = 'S_4d') -> list[Candidate]:
    """Build the initial list of Li candidates, one per ideal Wyckoff position.

    Each candidate inherits its label (by construction) and is associated with
    the nearest ideal cage centre, from which its ``cage_status`` is copied.
    """
    candidates: list[Candidate] = []
    for wyckoff in li_labels:
        for ideal_li in ideal_by_label[wyckoff]:
            j = nearest_cage_idx(ideal_li, cage_fracs)
            candidates.append(Candidate(
                wyckoff=wyckoff,
                full_label=wyckoff,
                cage_type=cage_type,
                cage_idx=j,
                cage_status=cage_status[j].status,
                ideal_frac=ideal_li.copy(),
                candidate_frac=(ideal_li + shift_frac) % 1.0,
            ))
    return candidates


# ---------------------------------------------------------------------------
# Iterative empirical-canonical learning
# ---------------------------------------------------------------------------

def reset_candidates_to_cif(candidates: list[Candidate],
                            ideal: dict[str, NDArray[np.float64]],
                            shift_frac: NDArray[np.float64],
                            li_labels: Iterable[str]) -> None:
    """Reset each candidate's ``ideal_frac`` / ``candidate_frac`` to the CIF start.

    Makes the iterative learning call idempotent on re-run: without this, a
    second execution would start from already-learned positions instead of
    the CIF ideals.
    """
    for lab in li_labels:
        cands_lab = [c for c in candidates if c.wyckoff == lab]
        assert len(cands_lab) == len(ideal[lab]), (
            f"{lab}: candidate count ({len(cands_lab)}) does not match "
            f"ideal count ({len(ideal[lab])})"
        )
        for c, ideal_pos in zip(cands_lab, ideal[lab]):
            c.ideal_frac     = ideal_pos.copy()
            c.candidate_frac = (ideal_pos + shift_frac) % 1.0


def learn_empirical_canonicals(
    candidates: list[Candidate],
    volume: DensityVolume,
    ideal: dict[str, NDArray[np.float64]],
    shift_frac: NDArray[np.float64],
    primitive_lattice: Lattice,
    supercell: tuple[int, int, int],
    li_labels: Iterable[str],
    *,
    refine_radius_A: float = 0.8,
    max_iter: int = 5,
    clean_disp_A: float = 0.55,
    converge_A: float = 0.05,
    runaway_A: float = 0.6,
    min_clean_for_learning: int = 10,
    verbose: bool = True,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
    """Iteratively refine each Wyckoff label's empirical canonical position.

    Each iteration:

    1. re-runs Pass-1-style local-max refinement on non-converged candidates;
    2. averages the "clean" refined positions (displacement below
       ``clean_disp_A``) after folding them onto the current empirical
       canonical with the space group's ops;
    3. projects onto the Wyckoff free-parameter surface and updates the
       canonical;
    4. re-seeds the candidates from the symmetry-expanded new canonical.

    Termination: when every label's canonical has moved less than
    ``converge_A`` per iteration, or ``max_iter`` is reached. Labels with
    fewer than ``min_clean_for_learning`` clean candidates, or a per-iter
    jump larger than ``runaway_A``, are frozen at their current value.

    Returns
    -------
    canonical_cif, empirical_canonical
        Two dicts keyed by Wyckoff label, mapping to primitive-cell
        fractional positions. ``canonical_cif`` is the initial (CIF-derived)
        value; ``empirical_canonical`` is the final learned value.
    """
    li_labels = tuple(li_labels)
    a_A = primitive_lattice.a

    canonical_cif       = {lab: project_wyckoff(super_to_prim(ideal[lab][0], supercell), lab)
                           for lab in li_labels}
    empirical_canonical = {lab: canonical_cif[lab].copy() for lab in li_labels}
    converged           = {lab: False for lab in li_labels}

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"ITERATIVE EMPIRICAL-CANONICAL LEARNING (max {max_iter} iter)")

    for it in range(1, max_iter + 1):
        # 1. Refine non-converged labels from their current candidate_frac
        for c in candidates:
            if converged[c.wyckoff]:
                continue
            refined, peak, disp = volume.local_max(c.candidate_frac, refine_radius_A)
            c.refined_frac      = refined
            c.refined_density   = peak
            c.refinement_disp_A = disp

        # 2. Update the empirical canonical from "clean" candidates per label
        log(f"\nIter {it}:")
        updated: list[str] = []
        for lab in li_labels:
            if converged[lab]:
                log(f"  {lab:>5}: (frozen)")
                continue

            clean = [c for c in candidates
                     if c.wyckoff == lab and c.refinement_disp_A < clean_disp_A]
            if len(clean) < min_clean_for_learning:
                log(f"  {lab:>5}: only {len(clean)} clean — freezing at current canonical")
                converged[lab] = True
                continue

            folded = np.array([
                fold_to_canonical(
                    super_to_prim((c.refined_frac - shift_frac) % 1.0, supercell),
                    empirical_canonical[lab],
                )
                for c in clean
            ])
            new_canon = project_wyckoff(circular_mean_frac(folded), lab)

            move_A = np.linalg.norm(pbc_mic(new_canon - empirical_canonical[lab])) * a_A
            if it > 1 and move_A > runaway_A:
                log(f"  {lab:>5}: ⚠ canonical jumped {move_A:.3f} Å — rejecting, freezing")
                converged[lab] = True
                continue

            empirical_canonical[lab] = new_canon
            updated.append(lab)

            cif_shift_A = np.linalg.norm(pbc_mic(new_canon - canonical_cif[lab])) * a_A
            log(f"  {lab:>5}: {len(clean):>2} clean, "
                f"Δ from CIF = {cif_shift_A:.3f} Å, moved {move_A:.3f} Å this iter")

            if move_A < converge_A:
                converged[lab] = True

        # 3. Re-seed candidate starts from the new empirical orbit
        for lab in updated:
            super_orbit = prim_to_super(expand_orbit(empirical_canonical[lab]), supercell)
            for cand in (cc for cc in candidates if cc.wyckoff == lab):
                diffs = pbc_mic(super_orbit - cand.ideal_frac)
                j = int(np.argmin(np.linalg.norm(diffs, axis=1)))
                cand.ideal_frac     = super_orbit[j]
                cand.candidate_frac = (super_orbit[j] + shift_frac) % 1.0

        if all(converged.values()):
            log(f"\n  ✓ all labels frozen at iter {it}")
            break
    else:
        log(f"\n  ⚠ iter budget exhausted without full convergence")

    return canonical_cif, empirical_canonical


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def find_duplicate_conflicts(candidates: list[Candidate],
                             lattice: Lattice,
                             threshold_A: float) -> list[tuple[int, int]]:
    """Flag pairs of candidates refined to within ``threshold_A`` of each other.

    Mutates each :class:`Candidate`'s ``conflicts_with`` list in place and
    returns the ``(i, j)`` pairs with ``i < j``.
    """
    coords = np.array([c.refined_frac for c in candidates])
    diffs = pbc_mic(coords[:, None, :] - coords[None, :, :])
    dists = np.linalg.norm(diffs @ np.asarray(lattice.matrix), axis=-1)

    conflict_pairs = [(int(i), int(j))
                      for i, j in np.argwhere(np.triu(dists < threshold_A, k=1))]

    for c in candidates:
        c.conflicts_with = []
    for i, j in conflict_pairs:
        candidates[i].conflicts_with.append(j)
        candidates[j].conflicts_with.append(i)
    return conflict_pairs


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_labeled_structure(candidates: list[Candidate], lattice: Lattice) -> Structure:
    """Build a :class:`~pymatgen.core.Structure` of Li atoms at the refined positions."""
    coords = np.array([c.refined_frac for c in candidates])
    labels = [c.full_label for c in candidates]
    return Structure(lattice=lattice, species=['Li'] * len(candidates),
                     coords=coords, labels=labels, coords_are_cartesian=False)


def _to_serialisable(value: Any) -> Any:
    """Recursively convert numpy types and nested containers to JSON-ready Python."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    return value


def export_metadata(path: str | Path,
                    *,
                    trajectory_path: str,
                    reference_cif: str,
                    output_cif: str,
                    supercell: tuple[int, int, int],
                    n_equilibration_steps: int,
                    aimd_lattice: Lattice,
                    shift_frac: NDArray[np.float64],
                    params: dict[str, float],
                    cage_4a_status: list[CageStatus],
                    cage_4d_status: list[CageStatus],
                    candidates: list[Candidate],
                    labeled: Structure) -> Path:
    """Write a JSON sidecar with all provenance and per-candidate data.

    Returns the path written.
    """
    metadata = {
        'timestamp':             datetime.now().isoformat(timespec='seconds'),
        'trajectory':            trajectory_path,
        'reference_cif':         reference_cif,
        'output_cif':            output_cif,
        'supercell':             list(supercell),
        'n_equilibration_steps': n_equilibration_steps,
        'aimd_lattice_abc':      [float(aimd_lattice.a),
                                  float(aimd_lattice.b),
                                  float(aimd_lattice.c)],
        'rigid_shift_frac':      shift_frac.tolist(),
        'params':                params,
        'cage_4a_status':        [_to_serialisable(asdict(c)) for c in cage_4a_status],
        'cage_4d_status':        [_to_serialisable(asdict(c)) for c in cage_4d_status],
        'candidates':            [_to_serialisable(asdict(c)) for c in candidates],
        'final_counts':          dict(Counter(s.label for s in labeled.sites)),
    }
    p = Path(path)
    with open(p, 'w') as f:
        json.dump(metadata, f, indent=2)
    return p
