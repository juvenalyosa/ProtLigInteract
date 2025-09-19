# Contributing to ProtLigInteract

Thanks for considering a contribution! This project is a PyMOL plugin focused on clear visuals, robust detection, and good UX. Small PRs are welcome.

## Dev setup

- Ensure you have a Qt-enabled PyMOL and Python 3.8+.
- Install dev deps into the Python used by PyMOL:

```
pip install -r requirements.txt
```

- Load the plugin in PyMOL via `Plugin > Plugin Manager > Install from local file` pointing to this repo folder or a zip archive.
- Modify `code_v2.py` or `protliginteract.ui` and reopen the dialog to see changes.

## Coding guidelines

- Keep changes focused and avoid unrelated refactors.
- Prefer explicit, readable code (no one-letter names).
- Catch PyMOL API calls in try/except where user scenes can vary.
- Group created objects per interaction type under `Interactions.<Type>`.
- Persist user-facing settings to `settings.json` with safe defaults.

## PR checklist

- Describe the change and its user impact.
- Test on at least one PDB with a ligand and one with a nucleotide.
- Verify PNG export (legend/scale/title) if visuals are affected.
- Update the README if behavior or setup changes.

## Reporting issues

Please include:
- PyMOL version, OS, Python version
- Example structure (PDB ID or file), steps taken
- Expected vs actual behavior and any errors

Thanks! Contributions help make the plugin better for everyone.

