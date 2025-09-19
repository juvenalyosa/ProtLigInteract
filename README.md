# ProtLigInteract — PyMOL Plugin for Protein–Ligand (and Nucleic Acid) Interactions

[![Sanity](https://github.com/OWNER/REPO/actions/workflows/sanity.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/sanity.yml)
[![Release](https://github.com/OWNER/REPO/actions/workflows/release.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/release.yml)

Note: replace OWNER/REPO above with your GitHub org/repo once published.

ProtLigInteract is a modern PyMOL plugin that detects and visualizes protein–ligand interactions (and protein–nucleic-acid contacts), with a fast, polished UI and export-ready scenes. It provides geometry-aware detection (H-bonds, halogens, cation–pi, pi–pi), cluster contacts (hydrophobic and van der Waals), ring-plane discs, per‑type styling, an in‑scene legend, and a powerful ligand chooser.

## Highlights

- Interaction detection
  - Hydrogen bonds with angle checks (D–H…A; fallback D…A–X)
  - Halogen bonds with C–X…A angle (σ‑hole direction)
  - Cation–pi, pi–pi (parallel and T‑shaped), metal coordination
  - Hydrophobic and van der Waals cluster contacts
- Accuracy & performance
  - Lightweight bond inference (covalent radii) for donors/acceptors
  - Ligand aromatic ring detection (planarity + 5/6‑member cycles)
  - Neighbor grid acceleration for fast proximity queries
  - mmCIF fallback and altloc (highest occupancy) handling
- Visuals
  - Per‑type color and dash styling; ring-plane CGO discs
  - Clouds for cluster contacts with opacity control
  - Angle labels (optional), in‑scene legend, optional scale bar + title
  - Grouped PyMOL objects per interaction type for easy show/hide
- UX tools
  - Filter table by type and max distance; CSV export
  - Ligand chooser with DNA/RNA support, proximity filtering, search
  - Live preview zoom in chooser; export ligand list to CSV
  - Settings panel with persistence; Redraw buttons; Apply Styles Now

## Requirements

- PyMOL 2.5+ (Qt-enabled builds)
- Python 3.8+
- NumPy
- Biopython (Bio.PDB)

Install missing Python deps into the interpreter used by PyMOL:

```
pip install numpy biopython
```

## Install (PyMOL Plugin Manager)

1. In PyMOL, open `Plugin > Plugin Manager > Install New Plugin > Install from local file`.
2. Download the latest `ProtLigInteract.zip` from the GitHub Releases page (preferred), or zip this folder yourself.
3. Restart PyMOL if prompted. The plugin appears under `Plugins > Protein-Ligand Interactions`.

Alternatively, you can copy/symlink this folder into your PyMOL plugins directory.

Releases
- Tagged pushes (`v*`) automatically publish a ready-to-install `ProtLigInteract.zip` asset.

Manual zip (fallback)
- From repo root, create a top-level folder in the zip (required by PyMOL Plugin Manager):
  - mkdir -p ProtLigInteract && cp __init__.py code_v2.py protliginteract.ui ProtLigInteract/
  - cp README.md requirements.txt ProtLigInteract/ 2>/dev/null || true
  - cp CONTRIBUTING.md ProtLigInteract/ 2>/dev/null || true
  - Optional: mkdir -p ProtLigInteract/screenshots && cp screenshots/*.png ProtLigInteract/screenshots/ 2>/dev/null || true
  - zip -r ProtLigInteract.zip ProtLigInteract

Scripted build (recommended for local zips)
- Use the included builder to create a correct zip with a top-level folder:

```
python tools/build_zip.py                 # builds dist/ProtLigInteract.zip
python tools/build_zip.py --name MyPlug   # custom folder name inside zip
python tools/build_zip.py --out out.zip   # custom output path
```

## Usage

1. Open the plugin: `Plugins > Protein-Ligand Interactions`.
2. Load a structure (PDB/mmCIF). The scene will show a white cartoon and auto-detect a ligand.
3. Optionally click `Choose…` to pick a ligand (includes DNA/RNA), with filters:
   - Show: All | Ligands | DNA/RNA
   - Proximity: Within X Å of Protein | Chain | Current Ligand
   - Search: chain/resn/resi/class (live filter)
   - Export CSV of the current list
4. Click `Calculate Interactions`.
5. Click rows in the table to toggle visuals. Use the left sidebar controls:
   - Filter by interaction type and max distance
   - Show Types checkboxes to toggle entire categories
   - Display Settings (legend, labels, ring discs, export options, styles)
   - Tools: Clear, Redraw, Redraw Visible Types, Apply Styles Now, Remove All Visuals
6. Use `Export PNG` for a high-res image with optional legend/scale bar/title.

## Screenshots

If you clone the repo, add images under `screenshots/` and they will render here. Example placeholders:

![Main](screenshots/main.png)
![Chooser](screenshots/chooser.png)
![Export](screenshots/export.png)

## Demo Video

[![Watch the demo](https://img.youtube.com/vi/ivQN7lLcBTU/0.jpg)](https://www.youtube.com/watch?v=ivQN7lLcBTU)

Click the thumbnail to watch a quick walkthrough of the plugin features and workflow on YouTube.

## Tips

- Angle labels are optional (Display Settings > Show Angle Labels).
- Per‑type style overrides (color, dash sizes, disc/cloud opacity, disc thickness) are in `Edit Styles…`.
- Legend shows category colors in-scene; toggle via Display Settings or by hiding `Interactions.Legend` group.
- The plugin stores preferences in `settings.json` (next to the plugin). Delete it to reset.

## Project Layout

- `__init__.py` — PyMOL plugin entrypoint
- `code_v2.py` — Main UI and interaction logic
- `protliginteract.ui` — Qt Designer UI file
- `settings.json` — Created at runtime to persist user preferences

## Development

- Open this repo in an environment with PyMOL available (or use PyMOL’s built-in Python).
- Install Python deps: `pip install -r requirements.txt`.
- Optional: set up pre-commit hooks for formatting/linting:

```
pip install pre-commit
pre-commit install
# Run once on the whole repo
pre-commit run --all-files
```
- Launch PyMOL and load the plugin via the Plugin Manager (Install from local file) pointing to the repo zip/folder.
- Edit `code_v2.py` / `protliginteract.ui`, then reopen the dialog from the menu to reload.

### Coding notes

- Keep UI changes in the `.ui` file when possible.
- Interaction detection should prefer geometric checks when affordable; use neighbor grids for performance.
- Visuals should be grouped under `Interactions.<Type>` to enable global toggles.

### Linting & formatting

- This repo includes Black and Ruff configuration (`pyproject.toml`) and a `.pre-commit-config.yaml`.
- Hooks will format code and fix/lint issues automatically on commit when installed.

## Troubleshooting

- Missing Biopython/NumPy: Install them into the Python used by PyMOL.
- No GUI: Ensure you’re using a Qt-enabled PyMOL build.
- Nothing detected: Check ligand selection; try `Choose…` and proximity filters.
- Exports missing legend/scale bar: Enable them in Display Settings.

## Roadmap ideas

- Optional RDKit integration for robust aromaticity/donor/acceptor perception
- KD‑tree backend for neighbor search when SciPy is present
- Batch analysis across multiple ligands with grouped outputs

## Acknowledgments

Built for researchers who need fast, clear structure interaction analysis and publication‑ready visuals, right inside PyMOL.
