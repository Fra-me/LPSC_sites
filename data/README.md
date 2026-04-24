# Test data

This folder contains a reduced gemdat trajectory cache and the reference CIF
used by the driver notebook:

- `reduced.cache` — gemdat cache of 5 000 post-equilibration AIMD steps for
  62 %-disorder O-doped Li₆PS₅Cl. Generated from a longer VASP run; the
  original `vasprun.xml` is not distributed here for space reasons.
- `LPSC_all.cif` — reference crystallographic structure used for the
  rigid-shift anchor and Wyckoff labels.

Both files are committed in the repo, so cloning is enough — no external
download needed.

Outputs written by the notebook (`sites_anchor.cif`, `sites_anchor.json`)
also land in this folder.