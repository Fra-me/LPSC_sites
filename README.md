# lpsc-sites

Anchor-based Li site extraction from AIMD density volumes for cubic argyrodite
Li₆PS₅Cl (F-43m).

## What it does

The framework sublattice (P, S, O, Cl) is static on the AIMD timescale
(MSD ≈ 0), so the Li Wyckoff site geometry is fixed relative to the framework.
This package places candidate Li sites at the expected Wyckoff positions,
rigidly aligns them to the AIMD frame via a P anchor, and refines each
candidate to the local density maximum with an iterative empirical-canonical
learning loop. Labels (48h, 48h′, 16e) attach by construction — no post-hoc
classification of watershed basins.

## Scope and assumptions

This is research code written around a specific material and method. The
implementation assumes:

- **Space group F-43m** (argyrodite family, cubic).
- **Wyckoff occupancy**: P at 4b, S/O at 4d, Cl at 4a, Li at 48h / 48h′ / 16e.
- **Disorder mode**: pairwise 4a ↔ 4d swaps between halogen and chalcogen.
- **Reference CIF labels**: `P1` (phosphorus), `S3` (4d sulfur), `Cl4` (4a
  chlorine), `48h` / `48h2` / `16e` (lithium). These are the labels used in
  the reference CIF of this project; adapt them in the notebook for another
  CIF convention.

The package is not a general Wyckoff-detection tool. For a different
argyrodite-family material with the same space group but different labels,
only a handful of constants at the top of the notebook need to change. For a
different space group, `F43M_OPS` in `lpsc_sites/symmetry.py` and the
`project_wyckoff` projections need revising.

## Installation

```bash
git clone https://github.com/TODO/lpsc-sites.git
cd lpsc-sites
pip install -e '.[dev]'
```

Python 3.10+ is required (uses `|`-style type unions and `slots=True`
dataclasses).

## Usage

The intended entry point is the driver notebook:

```bash
jupyter notebook notebooks/extract_sites.ipynb
```

Update the paths at the top of the notebook (`VASPRUN`, `CACHE`,
`REFERENCE_CIF`, `OUTPUT_CIF`) to point at your data, then run top to bottom.
Each cell either prints a short summary or produces a diagnostic plot; the
final cells export the labelled CIF plus a JSON metadata sidecar.

The same workflow can be driven from a plain Python script by importing
directly from the package — every pipeline step is a public function or
class. See the notebook for the canonical call order.

## Project layout

```
lpsc-sites/
├── lpsc_sites/              # installable package
│   ├── __init__.py          # public API
│   ├── records.py           # CageStatus, Candidate dataclasses
│   ├── geometry.py          # PBC primitives, rigid shift, supercell replication
│   ├── density.py           # DensityVolume class, refine_candidates
│   ├── symmetry.py          # F-43m ops, Wyckoff projection
│   └── pipeline.py          # cages, candidates, learning, conflicts, export
├── tests/                   # property tests for the mathematical invariants
│   └── test_lpsc_sites.py
├── notebooks/               # driver
│   └── extract_sites.ipynb
├── pyproject.toml
└── README.md
```

## Adapting to another material

If the space group is still F-43m and only the labels / disorder change,
editing the constants at the top of the notebook is enough:

| Constant            | What to change                                                              |
|---------------------|-----------------------------------------------------------------------------|
| `REFERENCE_CIF`     | Path to your reference CIF.                                                 |
| `P_LABEL`           | Label of the anchor species. Use a framework atom not involved in any swap.|
| `CL_LABEL`, `S_4D_LABEL` | Labels of the two framework sites involved in the swap.                |
| `HALOGENS`, `CHALCOGENS`  | Species allowed at the two swap sites.                                |
| `LI_LABELS`         | Wyckoff labels of the mobile-ion positions.                                |

If the Wyckoff projections differ, edit `project_wyckoff` in
`lpsc_sites/symmetry.py`. If the space group differs, replace `F43M_OPS`
and any callers that default to it.

## Running tests

```bash
pytest
```

15 property tests covering PBC handling, circular statistics, rigid-shift
recovery, Wyckoff projection, symmetry-orbit generation, and density-volume
queries. These are invariants — if any fails, extracted site positions are
unreliable.

## Method notes

Rationale for the key design choices:

- **Anchor via P (4b)**: 4b is not involved in the 4a/4d swap and is not a
  Li site, so the rigid shift it gives is not contaminated by disorder or
  mobile-ion drift.
- **Time-averaged framework**: absorbs real thermal distortions induced by
  disorder and oxygen substitution; distinguishes the method from a pure
  ideal-CIF overlay.
- **Two-pass refinement**: Pass 1 uses a wide ROI (~0.8 Å) to bridge the
  CIF ↔ AIMD mismatch; Pass 2 runs a tight ROI (~0.3 Å) from the learned
  canonical for sub-voxel precision.
- **Iterative canonical learning**: the CIF ideal is only an initial guess
  for the density peak. Averaging "clean" refinements under the orbit
  symmetry gives an empirical canonical that is then symmetry-expanded and
  used to re-seed the full candidate set.
- **Conflict detection via pairwise distance**: at N ≈ 224 candidates, a
  vectorised O(N²) distance matrix is simpler and faster than a KDTree, and
  handles the near-orthorhombic lattice without the `boxsize` approximation.

## License

MIT. See `LICENSE` (TODO).
