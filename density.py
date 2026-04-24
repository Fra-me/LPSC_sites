"""Density volume with PBC-aware local-max and integrated-density queries."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Lattice

from .geometry import pbc_mic
from .records import Candidate


class DensityVolume:
    """Scalar density volume on a regular fractional grid, with PBC.

    The volume spans one unit cell in fractional coordinates; voxel indices
    wrap with periodic boundary conditions. Queries take fractional
    coordinates and a cartesian radius (Å); the lattice metric is handled
    internally, so the class works for non-orthorhombic cells.

    Construct either directly from a numpy density and a lattice, or from a
    ``gemdat.Volume`` via :meth:`from_gemdat` (the latter preserves the
    source object so :meth:`plot_3d` can delegate to it).

    Parameters
    ----------
    density
        Scalar density sampled on a regular ``(Nx, Ny, Nz)`` grid.
    lattice
        pymatgen :class:`~pymatgen.core.Lattice` spanning one unit cell.
    source
        Optional underlying ``gemdat.Volume``. When supplied, :meth:`plot_3d`
        delegates to ``source.plot_3d``.
    """

    def __init__(self,
                 density: NDArray[np.float64],
                 lattice: Lattice,
                 source: object = None) -> None:
        self.density = np.asarray(density)
        self.lattice = lattice
        self.shape = np.array(self.density.shape)
        self.voxel_size_A = np.array([lattice.a, lattice.b, lattice.c]) / self.shape
        self._matrix = np.asarray(lattice.matrix)
        self._source = source

    @classmethod
    def from_gemdat(cls, gemdat_volume: object, lattice: Lattice) -> DensityVolume:
        """Construct from a ``gemdat.Volume``, retaining the source for plotting."""
        return cls(gemdat_volume.data, lattice, source=gemdat_volume)

    # ----- internal helpers -------------------------------------------------

    def _voxel_of(self, frac: NDArray[np.float64]) -> NDArray[np.int64]:
        idx = (np.asarray(frac) % 1.0 * self.shape).astype(int)
        return np.clip(idx, 0, self.shape - 1)

    def _box_around(self, frac: NDArray[np.float64], radius_A: float
                    ) -> tuple[NDArray[np.int64], NDArray[np.int64],
                               NDArray[np.int64], NDArray[np.float64]]:
        """Voxel indices ``(VI, VJ, VK)`` plus cartesian distances to ``frac``
        for every voxel in the bounding box that encloses the sphere.
        """
        half = np.ceil(radius_A / self.voxel_size_A).astype(int)
        ci = self._voxel_of(frac)
        Nx, Ny, Nz = self.shape
        di = np.arange(-half[0], half[0] + 1)
        dj = np.arange(-half[1], half[1] + 1)
        dk = np.arange(-half[2], half[2] + 1)
        DI, DJ, DK = np.meshgrid(di, dj, dk, indexing='ij')
        VI = (ci[0] + DI) % Nx
        VJ = (ci[1] + DJ) % Ny
        VK = (ci[2] + DK) % Nz
        fvox = np.stack([(VI + 0.5) / Nx,
                         (VJ + 0.5) / Ny,
                         (VK + 0.5) / Nz], axis=-1)
        dist = np.linalg.norm(pbc_mic(fvox - np.asarray(frac)) @ self._matrix, axis=-1)
        return VI, VJ, VK, dist

    # ----- public query API -------------------------------------------------

    def local_max(self, frac: NDArray[np.float64], radius_A: float
                  ) -> tuple[NDArray[np.float64], float, float]:
        """Voxel with the highest density within ``radius_A`` of ``frac``.

        Returns
        -------
        refined_frac, density_peak, displacement_A
            Fractional coordinates of the maximum-density voxel centre, the
            peak density value, and the distance (Å) from the input position
            to the refined one.
        """
        VI, VJ, VK, dist = self._box_around(frac, radius_A)
        vals = np.where(dist <= radius_A, self.density[VI, VJ, VK], -np.inf)
        i, j, k = np.unravel_index(int(np.argmax(vals)), vals.shape)
        best = np.array([VI[i, j, k], VJ[i, j, k], VK[i, j, k]])
        refined = (best + 0.5) / self.shape
        disp_A = float(np.linalg.norm(pbc_mic(refined - frac) @ self._matrix))
        return refined, float(self.density[tuple(best)]), disp_A

    def integrate(self, frac: NDArray[np.float64], radius_A: float) -> tuple[float, int]:
        """Sum the density over voxels whose centre lies within ``radius_A`` of ``frac``."""
        VI, VJ, VK, dist = self._box_around(frac, radius_A)
        mask = dist <= radius_A
        return float(self.density[VI[mask], VJ[mask], VK[mask]].sum()), int(mask.sum())

    # ----- delegated visualisation -----------------------------------------

    def plot_3d(self, **kwargs):
        """Delegate 3D visualisation to the underlying ``gemdat.Volume``.

        Only works when the :class:`DensityVolume` was constructed with a
        source (e.g. via :meth:`from_gemdat`). All keyword arguments are
        forwarded unchanged.
        """
        if self._source is None:
            raise RuntimeError(
                "plot_3d requires a gemdat.Volume source; construct the "
                "DensityVolume via DensityVolume.from_gemdat(...)."
            )
        return self._source.plot_3d(**kwargs)


def refine_candidates(candidates: list[Candidate],
                      volume: DensityVolume,
                      refine_radius_A: float,
                      integrate_radius_A: float | None = None) -> None:
    """Refine each candidate's ``candidate_frac`` to the nearest local density maximum.

    Mutates the :class:`Candidate` objects in place: populates
    ``refined_frac``, ``refined_density``, ``refinement_disp_A`` and — if
    ``integrate_radius_A`` is given — ``integrated_density`` and
    ``integrated_voxels``.
    """
    for c in candidates:
        refined, peak, disp = volume.local_max(c.candidate_frac, refine_radius_A)
        c.refined_frac      = refined
        c.refined_density   = peak
        c.refinement_disp_A = disp
        if integrate_radius_A is not None:
            total, n_vox = volume.integrate(refined, integrate_radius_A)
            c.integrated_density = total
            c.integrated_voxels  = n_vox
