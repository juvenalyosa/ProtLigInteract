# ProtLigInteract/code_v2.py
import os
import sys
import math
import collections
from collections import defaultdict

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from pymol import cgo, cmd
from pymol.Qt import QtCore, QtWidgets
from pymol.Qt.utils import loadUi

try:
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    HAS_RDKIT_MPL = True
except ImportError:
    HAS_RDKIT_MPL = False

# --- Expanded color palette for new interaction types ---
COLOR_MAP = {
    "Hydrogen Bond": "#2CF6F3",     # Cyan-ish
    "Salt Bridge": "#F64D4D",       # Soft Red
    "Hydrophobic": "#55E089",       # Soft Green
    "Pi-Pi Stacking": "#F6A83E",    # Soft Orange
    "T-Shaped Pi-Pi": "#F68E28",    # Darker Org
    "Cation-Pi": "#F63EF6",         # Magenta
    "Anion-Pi": "#FF4444",          # Red
    "Halogen Bond": "#E0E055",      # Yellow-ish
    "Metal Coordination": "#A83EF6",# Purple
    "Van der Waals": "#F6C870",     # Light Orange
    "Selected": "#FFFF55",          # Bright Yellow
}

# Per-type dash styling (applied per distance object)
STYLE_MAP = {
    "Hydrogen Bond": {"dash_radius": 0.06, "dash_length": 0.35, "dash_gap": 0.15},
    "Halogen Bond": {"dash_radius": 0.07, "dash_length": 0.40, "dash_gap": 0.18},
    "Salt Bridge": {"dash_radius": 0.09, "dash_length": 0.55, "dash_gap": 0.22},
    "Metal Coordination": {"dash_radius": 0.07, "dash_length": 0.45, "dash_gap": 0.18},
    "Cation-Pi": {"dash_radius": 0.08, "dash_length": 0.45, "dash_gap": 0.20},
    "Anion-Pi": {"dash_radius": 0.08, "dash_length": 0.45, "dash_gap": 0.20},
    "Pi-Pi Stacking": {"dash_radius": 0.08, "dash_length": 0.45, "dash_gap": 0.20},
    "T-Shaped Pi-Pi": {"dash_radius": 0.08, "dash_length": 0.45, "dash_gap": 0.20},
}

# runtime color cache for overrides
_COLOR_CACHE = {}


def _hex_to_rgb(hexstr):
    s = hexstr.strip().lstrip("#")
    if len(s) != 6:
        return None
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


# --- Updated Geometric and Chemical Definitions based on user rules ---
GEOMETRY_CRITERIA = {
    "h_bond_dist": 3.5,
    "h_bond_angle": 120,
    "salt_bridge_dist": 4.0,
    "hydrophobic_dist": 4.0,
    "pi_pi_dist": 5.0,  # Centroid distance for parallel
    "pi_pi_angle": 30,  # Angle between normals for parallel
    "pi_t_dist": 6.0,  # Centroid distance for T-shaped
    "pi_t_angle_low": 60,  # Lower angle for T-shaped
    "pi_t_angle_high": 90,  # Upper angle for T-shaped
    "cation_pi_dist": 6.0,
    "cation_pi_angle": 30,  # Angle between normal and cation vector
    "halogen_dist": 3.5,
    "halogen_angle": 150,  # C-X...Y angle
    "metal_coordination_dist": 3.0,
    "vdw_dist": 4.0,
}

# Definitions for charged, donor, acceptor, and aromatic atoms
ATOM_DEFS = {
    "protein_positive": {"LYS": ["NZ"], "ARG": ["NH1", "NH2"], "HIS": ["ND1", "NE2"]},
    "protein_negative": {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]},
    "protein_h_donors": {
        "main": ["N"],
        "LYS": ["NZ"],
        "ARG": ["NE", "NH1", "NH2"],
        "HIS": ["ND1", "NE2"],
        "TRP": ["NE1"],
        "GLN": ["NE2"],
        "ASN": ["ND2"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "TYR": ["OH"],
    },
    "protein_h_acceptors": {
        "main": ["O"],
        "ASP": ["OD1", "OD2"],
        "GLU": ["OE1", "OE2"],
        "HIS": ["ND1", "NE2"],
        "GLN": ["OE1"],
        "ASN": ["OD1"],
        "SER": ["OG"],
        "THR": ["OG1"],
        "TYR": ["OH"],
    },
    "protein_aromatic_rings": {
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    },
    "protein_hydrophobic_res": ["ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TRP", "MET"],
    "halogens": ["F", "CL", "BR", "I"],
    "metals": ["ZN", "MG", "CA", "FE", "MN", "CU", "NI", "CO", "K", "NA"],
}

# Common PDB residue names for nucleotides (DNA/RNA)
NUCLEOTIDE_RESN = set(["A", "C", "G", "U", "DA", "DC", "DG", "DT", "I", "DI"])

# Simple covalent radii (Å) for bond estimation and angle geometry
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
    "S": 1.05,
    "P": 1.07,
}
BONDTOL = 0.45  # soft tolerance added to radii sum


def distance(a, b):
    """Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(a) - np.array(b))


def get_centroid(coords):
    """Compute centroid of a list of 3D coordinates."""
    return np.mean(np.array(coords), axis=0)


def get_ring_radius(coords, centroid=None):
    c = centroid if centroid is not None else get_centroid(coords)
    return float(max(distance(p, c) for p in coords))


def get_normal(coords):
    """Compute normal vector of a ring plane defined by coords."""
    pts = np.array(coords)
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0])
    centroid = get_centroid(pts)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    n = np.linalg.norm(normal)
    if n == 0:
        return np.array([0.0, 0.0, 1.0])
    return normal / n


def angle_between_vectors(v1, v2):
    """Calculate the angle in degrees between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    v1_u = v1 / n1
    v2_u = v2 / n2
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


class ProtLigInteractDialog(QtWidgets.QDialog):
    """Main dialog for the Protein-Ligand Interaction Visualizer plugin."""

    def __init__(self, parent=None):
        super().__init__(parent)
        uifile = os.path.join(os.path.dirname(__file__), "protliginteract.ui")
        loadUi(uifile, self)
        self.setWindowTitle("Protein-Ligand Interaction Visualizer")

        self.pdb_path = None
        self.structure = None
        self.loaded_object = None
        self.ligand_info = None
        self.interactions = []
        self._selected_ids = set()
        self._table_items = []
        self._connect_signals()
        self._setup_trajectory_ui()

    def _connect_signals(self):
        self.file_btn.clicked.connect(self._on_load_pdb)
        self.detect_ligand_btn.clicked.connect(self._on_detect_ligand)
        if hasattr(self, "choose_ligand_btn"):
            self.choose_ligand_btn.clicked.connect(self._on_choose_ligand)
        self.calc_btn.clicked.connect(self._on_calc_interactions)
        self.table.cellClicked.connect(self._on_table_clicked)
        self.render_btn.clicked.connect(self._on_render_export)
        # Optional tools if present in UI
        if hasattr(self, "clear_btn"):
            self.clear_btn.clicked.connect(self._on_clear)
        if hasattr(self, "export_csv_btn"):
            self.export_csv_btn.clicked.connect(self._on_export_csv)
        if hasattr(self, "redraw_btn"):
            self.redraw_btn.clicked.connect(self._on_redraw_all)
        if hasattr(self, "redraw_visible_btn"):
            self.redraw_visible_btn.clicked.connect(self._on_redraw_visible_types)
        if hasattr(self, "reset_defaults_btn"):
            self.reset_defaults_btn.clicked.connect(self._on_reset_defaults)
        if hasattr(self, "hist_btn"):
            self.hist_btn.setVisible(False) # User request: move functionality to Trajectory only
            # self.hist_btn.clicked.connect(self._on_show_hist)
        if hasattr(self, "edit_styles_btn"):
            self.edit_styles_btn.clicked.connect(self._on_edit_styles)
        if hasattr(self, "apply_styles_btn"):
            self.apply_styles_btn.clicked.connect(self._on_apply_styles_now)
        if hasattr(self, "remove_all_btn"):
            self.remove_all_btn.clicked.connect(self._on_remove_all_visuals)

        # Add 2D Map button programmatically if not in UI
        if not hasattr(self, "show_2d_btn"):
            self.show_2d_btn = QtWidgets.QPushButton("Show 2D Map")
            self.show_2d_btn.clicked.connect(self._on_show_2d_map)
            # Try to add to the tools layout or near render button
            # Assuming 'tools_group' or similar exists, or add to main layout
            # For robustness, we'll try to find a suitable layout
            if hasattr(self, "verticalLayout"): # Common top layout name
                self.verticalLayout.addWidget(self.show_2d_btn)
            else:
                self.layout().addWidget(self.show_2d_btn)
            self.show_2d_btn.setEnabled(False) # Enable after calculation

    def _on_load_pdb(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open PDB File", "", "PDB/CIF Files (*.pdb *.ent *.cif)")
        if not path:
            return
        self.pdb_path = path
        if hasattr(self, "file_edit"):
            self.file_edit.setText(path)
        try:
            self._load_structure(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to parse structure: {e}")
            return
        base = os.path.splitext(os.path.basename(path))[0]
        cmd.reinitialize()
        self.loaded_object = base
        try:
            cmd.load(path, base)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load in PyMOL: {e}")
            return
        # persist last file
        try:
            self._settings["last_file"] = path
            self._save_settings()
        except Exception:
            pass
        ligand = self._auto_detect_ligand()
        if ligand:
            self.ligand_info = ligand
            self.ligand_chain.setText(ligand[0])
            self.ligand_resi.setText(str(ligand[1]))
            self.ligand_resn.setText(ligand[2])
            # persist last ligand
            try:
                self._settings.update(
                    {"last_ligand_chain": ligand[0], "last_ligand_resi": int(ligand[1]), "last_ligand_resn": ligand[2]}
                )
                self._save_settings()
            except Exception:
                pass
        self._beautify_scene()
        self.interactions.clear()
        self.table.setRowCount(0)
        self._selected_ids.clear()

    def _load_structure(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("X", path)

    def _on_detect_ligand(self):
        if not self.structure:
            QtWidgets.QMessageBox.warning(self, "Error", "Load a PDB file first.")
            return
        ligand = self._auto_detect_ligand()
        if not ligand:
            QtWidgets.QMessageBox.warning(self, "Error", "No ligand detected.")
            return
        self.ligand_info = ligand
        self.ligand_chain.setText(ligand[0])
        self.ligand_resi.setText(str(ligand[1]))
        self.ligand_resn.setText(ligand[2])
        try:
            self._settings.update(
                {"last_ligand_chain": ligand[0], "last_ligand_resi": int(ligand[1]), "last_ligand_resn": ligand[2]}
            )
            self._save_settings()
        except Exception:
            pass
        self._beautify_scene()

    def _auto_detect_ligand(self):
        max_atoms = 0
        best = None
        for res in self.structure.get_residues():
            if not is_aa(res) and res.get_resname() not in ATOM_DEFS["metals"] and res.get_resname() != "HOH":
                heavy_atoms = [a for a in res if a.element != "H"]
                if len(heavy_atoms) > max_atoms:
                    max_atoms = len(heavy_atoms)
                    best = (res.parent.id, res.id[1], res.get_resname())
        return best

    def _discover_ligands(self):
        # Return a list of potential ligands with metadata
        ligands = []
        if not self.structure:
            return ligands
        seen = set()
        for res in self.structure.get_residues():
            resn = res.get_resname()
            if is_aa(res) or resn in ATOM_DEFS["metals"] or resn == "HOH":
                continue
            chain_id = res.parent.id
            resi = res.id[1]
            key = (chain_id, resi, resn)
            if key in seen:
                continue
            seen.add(key)
            heavy_atoms = [a for a in res if getattr(a, "element", "") != "H"]
            ligands.append(
                {
                    "chain": chain_id,
                    "resi": int(resi),
                    "resn": resn,
                    "heavy_count": len(heavy_atoms),
                    "class": "nucleotide" if resn.upper() in NUCLEOTIDE_RESN else "small_molecule",
                }
            )
        # Sort nucleotides first, then by heavy atoms desc, then chain/resi
        ligands.sort(key=lambda x: (0 if x["class"] == "nucleotide" else 1, -x["heavy_count"], x["chain"], x["resi"]))
        return ligands

    def _on_choose_ligand(self):
        if not self.structure:
            QtWidgets.QMessageBox.warning(self, "Error", "Load a structure first.")
            return
        items = self._discover_ligands()
        if not items:
            QtWidgets.QMessageBox.information(
                self, "No ligands", "No non-polymer, non-water, non-metal residues detected."
            )
            return
        # Build chooser dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Choose Ligand")
        v = QtWidgets.QVBoxLayout(dlg)
        # Filter row
        filt_row = QtWidgets.QHBoxLayout()
        filt_row.addWidget(QtWidgets.QLabel("Show:"))
        filt_combo = QtWidgets.QComboBox(dlg)
        filt_combo.addItems(["All", "Ligands", "DNA/RNA"])
        filt_row.addSpacing(12)
        near_cb = QtWidgets.QCheckBox("Within")
        near_spin = QtWidgets.QDoubleSpinBox()
        near_spin.setDecimals(1)
        near_spin.setRange(1.0, 20.0)
        near_spin.setSingleStep(0.5)
        near_spin.setValue(6.0)
        near_cb.setChecked(False)
        unit_lbl = QtWidgets.QLabel("Å of")
        ref_combo = QtWidgets.QComboBox(dlg)
        ref_combo.addItem("Protein")
        # populate chains
        chains = []
        try:
            chains = sorted({ch.id for ch in self.structure.get_chains()})
        except Exception:
            chains = []
        ref_combo.addItem("Chain")
        if self.ligand_info:
            ref_combo.addItem("Current Ligand")
        chain_combo = QtWidgets.QComboBox(dlg)
        chain_combo.addItems([str(c) for c in chains] if chains else [])
        chain_combo.setEnabled(False)
        near_spin.setEnabled(False)

        def on_near_toggle(on):
            near_spin.setEnabled(on)
            unit_lbl.setEnabled(on)
            ref_combo.setEnabled(on)
            chain_combo.setEnabled(on and ref_combo.currentText() == "Chain")

        near_cb.toggled.connect(on_near_toggle)

        def on_ref_change(_):
            chain_combo.setEnabled(near_cb.isChecked() and ref_combo.currentText() == "Chain")

        ref_combo.currentTextChanged.connect(on_ref_change)
        for w in (near_cb, near_spin, unit_lbl, ref_combo, chain_combo):
            filt_row.addWidget(w)
        # Search box
        filt_row.addSpacing(12)
        search_edit = QtWidgets.QLineEdit(dlg)
        search_edit.setPlaceholderText("Search (chain/resn/resi)")
        filt_row.addWidget(search_edit)
        filt_row.addStretch(1)
        v.addLayout(filt_row)

        table = QtWidgets.QTableWidget(dlg)
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["Chain", "Resi", "Resn", "Heavy atoms", "Class", "Min Dist (Å)"])
        table.setSortingEnabled(True)

        def populate(rows):
            table.setRowCount(0)
            table.setRowCount(len(rows))
            for i, it in enumerate(rows):
                table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(it["chain"])))
                table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(it["resi"])))
                table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(it["resn"])))
                table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(it["heavy_count"])))
                table.setItem(i, 4, QtWidgets.QTableWidgetItem("DNA/RNA" if it["class"] == "nucleotide" else "Ligand"))
                # Min distance to current reference when proximity filter is enabled
                mind_txt = ""
                if near_cb.isChecked():
                    md = min_dist_reference(it)
                    if md is not None:
                        mind_txt = f"{md:.2f}"
                item = QtWidgets.QTableWidgetItem(mind_txt)
                # Make numeric sort work even if empty by setting data role
                try:
                    item.setData(QtCore.Qt.UserRole, float(mind_txt) if mind_txt else 1e9)
                except Exception:
                    pass
                table.setItem(i, 5, item)
            table.resizeColumnsToContents()

        items_full = items[:]
        populate(items_full)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        v.addWidget(table)
        # Preview checkbox and behavior
        preview_row = QtWidgets.QHBoxLayout()
        preview_cb = QtWidgets.QCheckBox("Preview zoom on hover/selection", dlg)
        preview_cb.setChecked(True)
        preview_row.addWidget(preview_cb)
        preview_row.addStretch(1)
        v.addLayout(preview_row)

        # Build reference coordinate caches
        prot_coords = []
        chain_coords = {c: [] for c in chains}
        try:
            for res in self.structure.get_residues():
                ch = res.parent.id
                # protein heavy atoms for protein reference
                if is_aa(res):
                    for a in res:
                        if getattr(a, "element", "") != "H":
                            prot_coords.append(np.array(a.coord))
                # chain coords: include all heavy atoms (protein + nucleic acid + others)
                if ch in chain_coords:
                    for a in res:
                        if getattr(a, "element", "") != "H":
                            chain_coords[ch].append(np.array(a.coord))
            prot_coords = np.array(prot_coords) if prot_coords else np.zeros((0, 3))
            for ch in list(chain_coords.keys()):
                arr = np.array(chain_coords[ch]) if chain_coords[ch] else np.zeros((0, 3))
                chain_coords[ch] = arr
        except Exception:
            prot_coords = np.zeros((0, 3))
            chain_coords = {c: np.zeros((0, 3)) for c in chains}

        def residue_atoms(chain, resi, resn):
            try:
                for res in self.structure.get_residues():
                    if res.parent.id == chain and res.id[1] == resi and res.get_resname() == resn:
                        return [np.array(a.coord) for a in res if getattr(a, "element", "") != "H"]
            except Exception:
                pass
            return []

        def near_reference(it, thr):
            # Determine reference coords
            ref = ref_combo.currentText()
            if ref == "Protein":
                rc = prot_coords
            elif ref == "Chain":
                ch = chain_combo.currentText()
                rc = chain_coords.get(ch, np.zeros((0, 3)))
            else:  # Current Ligand
                rc_list = []
                try:
                    if self.ligand_info:
                        c, i, r = self.ligand_info
                        rc_list = residue_atoms(c, int(i), r)
                except Exception:
                    rc_list = []
                rc = np.array(rc_list) if rc_list else np.zeros((0, 3))
            if rc.shape[0] == 0:
                return False
            lig = residue_atoms(it["chain"], it["resi"], it["resn"])
            if not lig:
                return False
            for la in lig:
                d = np.linalg.norm(rc - la, axis=1).min()
                if d <= thr:
                    return True
            return False

        def min_dist_reference(it):
            # Returns minimal heavy-atom distance to current reference or None
            ref = ref_combo.currentText()
            if ref == "Protein":
                rc = prot_coords
            elif ref == "Chain":
                ch = chain_combo.currentText()
                rc = chain_coords.get(ch, np.zeros((0, 3)))
            else:
                rc_list = []
                try:
                    if self.ligand_info:
                        c, i, r = self.ligand_info
                        rc_list = residue_atoms(c, int(i), r)
                except Exception:
                    rc_list = []
                rc = np.array(rc_list) if rc_list else np.zeros((0, 3))
            if rc.shape[0] == 0:
                return None
            lig = residue_atoms(it["chain"], it["resi"], it["resn"])
            if not lig:
                return None
            best = None
            for la in lig:
                d = float(np.linalg.norm(rc - la, axis=1).min())
                if best is None or d < best:
                    best = d
            return best

        def apply_filter():
            mode = filt_combo.currentText()
            if mode == "All":
                rows = items_full
            elif mode == "Ligands":
                rows = [x for x in items_full if x["class"] == "small_molecule"]
            else:
                rows = [x for x in items_full if x["class"] == "nucleotide"]
            if near_cb.isChecked():
                thr = float(near_spin.value())
                rows = [x for x in rows if near_reference(x, thr)]
            q = search_edit.text().strip().lower()
            if q:

                def match(rec):
                    blob = f"{rec['chain']} {rec['resn']} {rec['resi']} {rec['class']}".lower()
                    return q in blob

                rows = [r for r in rows if match(r)]
            populate(rows)
            try:
                if near_cb.isChecked():
                    table.sortItems(5, QtCore.Qt.AscendingOrder)
            except Exception:
                pass

        filt_combo.currentTextChanged.connect(lambda _t: (apply_filter(), _save_chooser_settings()))
        near_cb.toggled.connect(lambda _on: (apply_filter(), _save_chooser_settings()))
        near_spin.valueChanged.connect(lambda _v: (apply_filter(), _save_chooser_settings()))
        ref_combo.currentTextChanged.connect(lambda _t: (apply_filter(), _save_chooser_settings()))
        chain_combo.currentTextChanged.connect(lambda _t: (apply_filter(), _save_chooser_settings()))
        search_edit.textChanged.connect(lambda _t: (apply_filter(), _save_chooser_settings()))

        def zoom_to_row(row):
            if row < 0 or row >= table.rowCount():
                return
            if not preview_cb.isChecked():
                return
            try:
                ch = table.item(row, 0).text()
                ri = table.item(row, 1).text()
                rn = table.item(row, 2).text()
                sel = f"{self.loaded_object} and chain {ch} and resi {ri} and resn {rn}"
                cmd.zoom(sel, 8)
            except Exception:
                pass

        table.setMouseTracking(True)
        table.cellEntered.connect(lambda r, c: zoom_to_row(r))
        table.itemSelectionChanged.connect(lambda: zoom_to_row(table.currentRow()))

        # Restore chooser settings
        try:
            cs = self._settings
            # show mode
            if cs.get("chooser_show") in ["All", "Ligands", "DNA/RNA"]:
                filt_combo.setCurrentText(cs.get("chooser_show"))
            # near filter
            near_cb.setChecked(bool(cs.get("chooser_near", False)))
            near_spin.setValue(float(cs.get("chooser_near_dist", 6.0)))
            if cs.get("chooser_ref") in ["Protein", "Chain", "Current Ligand"]:
                ref_combo.setCurrentText(cs.get("chooser_ref"))
            # chain selection
            ch_pref = cs.get("chooser_ref_chain", "")
            if ch_pref and ch_pref in [chain_combo.itemText(i) for i in range(chain_combo.count())]:
                chain_combo.setCurrentText(ch_pref)
            search_edit.setText(cs.get("chooser_search", ""))
            apply_filter()
        except Exception:
            apply_filter()
        h = QtWidgets.QHBoxLayout()
        export_btn = QtWidgets.QPushButton("Export CSV", dlg)
        select_btn = QtWidgets.QPushButton("Select", dlg)
        cancel_btn = QtWidgets.QPushButton("Cancel", dlg)
        h.addWidget(export_btn)
        h.addStretch(1)
        h.addWidget(select_btn)
        h.addWidget(cancel_btn)
        v.addLayout(h)

        def do_select():
            idx = table.currentRow()
            if idx < 0:
                QtWidgets.QMessageBox.warning(dlg, "Select", "Please select a row.")
                return
            # read back from table to find selected ligand
            chain = table.item(idx, 0).text()
            resi = int(table.item(idx, 1).text())
            resn = table.item(idx, 2).text()
            it = next(
                (
                    x
                    for x in items_full
                    if str(x["chain"]) == chain and int(x["resi"]) == resi and str(x["resn"]) == resn
                ),
                None,
            )
            if it is None:
                QtWidgets.QMessageBox.warning(dlg, "Select", "Could not resolve selection.")
                return
            self.ligand_info = (it["chain"], int(it["resi"]), it["resn"])
            self.ligand_chain.setText(str(it["chain"]))
            self.ligand_resi.setText(str(it["resi"]))
            self.ligand_resn.setText(str(it["resn"]))
            try:
                self._settings.update(
                    {
                        "last_ligand_chain": it["chain"],
                        "last_ligand_resi": int(it["resi"]),
                        "last_ligand_resn": it["resn"],
                    }
                )
                self._save_settings()
            except Exception:
                pass
            dlg.accept()

        select_btn.clicked.connect(do_select)
        cancel_btn.clicked.connect(dlg.reject)
        table.itemDoubleClicked.connect(lambda *_: do_select())

        def do_export():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Export Ligands CSV", "ligands.csv", "CSV (*.csv)")
            if not path:
                return
            try:
                import csv

                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    headers = ["Chain", "Resi", "Resn", "Heavy atoms", "Class"]
                    if table.columnCount() >= 6:
                        headers.append("Min Dist (Å)")
                    w.writerow(headers)
                    for row in range(table.rowCount()):
                        vals = [
                            table.item(row, 0).text() if table.item(row, 0) else "",
                            table.item(row, 1).text() if table.item(row, 1) else "",
                            table.item(row, 2).text() if table.item(row, 2) else "",
                            table.item(row, 3).text() if table.item(row, 3) else "",
                            table.item(row, 4).text() if table.item(row, 4) else "",
                        ]
                        if table.columnCount() >= 6:
                            vals.append(table.item(row, 5).text() if table.item(row, 5) else "")
                        w.writerow(vals)
                QtWidgets.QMessageBox.information(dlg, "Exported", f"Ligand list saved to {path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Error", f"Failed to export CSV: {e}")

        export_btn.clicked.connect(do_export)

        def _save_chooser_settings():
            try:
                self._settings.update(
                    {
                        "chooser_show": filt_combo.currentText(),
                        "chooser_near": bool(near_cb.isChecked()),
                        "chooser_near_dist": float(near_spin.value()),
                        "chooser_ref": ref_combo.currentText(),
                        "chooser_ref_chain": (
                            chain_combo.currentText() if chain_combo.isEnabled() and chain_combo.count() > 0 else ""
                        ),
                        "chooser_search": search_edit.text(),
                    }
                )
                self._save_settings()
            except Exception:
                pass

        dlg.resize(540, 360)
        dlg.exec_()

    def _on_calc_interactions(self):
        if not self.ligand_chain.text().strip():
            QtWidgets.QMessageBox.warning(self, "Error", "Specify ligand info.")
            return
        self.ligand_info = (
            self.ligand_chain.text().strip(),
            int(self.ligand_resi.text().strip()),
            self.ligand_resn.text().strip(),
        )
        
        # Validate Ligand Hydrogens for Charge Calculation
        self._target_ph = None
        try:
             # Find ligand atoms in Bio.PDB structure
             # Assuming structure is loaded and matched
             l_c, l_i, l_n = self.ligand_info
             has_ligand_h = False
             
             # Locate residue
             target_res = None
             for model in self.structure:
                 for chain in model:
                     if chain.id == l_c:
                         # Robust Residue Lookup (handling Bio.PDB tuple keys)
                         # Key format: (hetero_flag, sequence_id, insertion_code)
                         # We match sequence_id == int(l_i) and resname == l_n
                         found_res = None
                         for r in chain:
                             if r.id[1] == int(l_i) and r.get_resname() == l_n:
                                 found_res = r
                                 break
                         
                         if found_res:
                             target_res = found_res
                             break
                 if target_res: break
             
             if target_res:
                  if any(a.element == "H" for a in target_res):
                       has_ligand_h = True
                  
                  if not has_ligand_h:
                       # User Requirement: Pop up window asking for pH
                       ph, ok = QtWidgets.QInputDialog.getDouble(
                            self, "Protonation Required", 
                            "The ligand has no hydrogens.\n"
                            "To correctly verify charges (Anion/Cation-Pi, Salt Bridges),\n"
                            "please enter the system pH for protonation:",
                            7.4, 0.0, 14.0, 1
                       )
                       if ok:
                            self._target_ph = ph
                            # Also check protein hydrogens?
                            # If typical protein is missing H, add them (standard pH)
                            sel_h = f"({self.loaded_object} and elem H)"
                            n_h = cmd.count_atoms(sel_h)
                            n_all = cmd.count_atoms(f"({self.loaded_object})")
                            if n_all > 0 and (n_h / n_all) < 0.1:
                                 reply = QtWidgets.QMessageBox.question(
                                     self, "Add Hydrogens to Protein?",
                                     "Protein also seems to lack hydrogens.\n"
                                     "Add standard hydrogens to protein (Glu, Asp, Lys, Arg, His charges)?",
                                     QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                                 )
                                 if reply == QtWidgets.QMessageBox.Yes:
                                      cmd.h_add(self.loaded_object)
                       else:
                            # User cancelled
                            return
             
        except Exception as e:
             print(f"Pre-calc Check Error: {e}")
             
        self.interactions = self._calculate_all_interactions()
        self._populate_table()
        self._beautify_scene()
        
        # Calculate SASA
        sasa_info = self._calc_sasa()
        msg_extra = ""
        if sasa_info:
            msg_extra = (
                f"\n\nSASA (Ligand):\n"
                f"  Bound: {sasa_info['bound']:.1f} Å²\n"
                f"  Free:  {sasa_info['free']:.1f} Å²\n"
                f"  Buried: {sasa_info['buried']:.1f} Å² ({sasa_info['buried_pct']:.1f}%)"
            )
        
        # Enable 2D map button if requirements met
        if hasattr(self, "show_2d_btn"):
             self.show_2d_btn.setEnabled(HAS_RDKIT_MPL)
             
        QtWidgets.QMessageBox.information(self, "Done", f"Found {len(self.interactions)} interactions.{msg_extra}")
        # Ensure toggles reflect group visibility
        try:
            for t, cb in getattr(self, "_type_checkboxes", {}).items():
                self._on_type_toggle(t, cb.isChecked())
        except Exception:
            pass
        # persist ligand fields
        try:
            self._settings.update(
                {
                    "last_ligand_chain": self.ligand_info[0],
                    "last_ligand_resi": int(self.ligand_info[1]),
                    "last_ligand_resn": self.ligand_info[2],
                }
            )
            self._save_settings()
        except Exception:
            pass

    def _setup_trajectory_ui(self):
        # Programmatically add Trajectory Analysis UI to the layout
        # Assumes a vertical layout exists. We can add a GroupBox at the bottom.
        try:
             # Find main layout. usually self.layout() or self.centralWidget().layout()
             layout = self.layout()
             
             gb = QtWidgets.QGroupBox("Trajectory Analysis")
             vbox = QtWidgets.QVBoxLayout()
             
             # File Loaders
             hbox_load = QtWidgets.QHBoxLayout()
             self.btn_load_top = QtWidgets.QPushButton("Load Topology")
             self.btn_load_top.setToolTip("Load PDB, PSF, GRO, etc.")
             self.btn_load_top.clicked.connect(self._on_load_topology)
             hbox_load.addWidget(self.btn_load_top)
             
             self.btn_load_traj = QtWidgets.QPushButton("Load Trajectory")
             self.btn_load_traj.setToolTip("Load DCD, XTC, TRR, etc. into current object")
             self.btn_load_traj.clicked.connect(self._on_load_trajectory)
             hbox_load.addWidget(self.btn_load_traj)
             
             vbox.addLayout(hbox_load)
             
             # Analysis Controls
             hbox = QtWidgets.QHBoxLayout()
             hbox.addWidget(QtWidgets.QLabel("Step (Frames):"))
             self.traj_step = QtWidgets.QSpinBox()
             self.traj_step.setRange(1, 1000)
             self.traj_step.setValue(1)
             hbox.addWidget(self.traj_step)
             
             self.btn_traj = QtWidgets.QPushButton("Run Analysis (Histogram)")
             self.btn_traj.clicked.connect(self._run_trajectory_analysis)
             hbox.addWidget(self.btn_traj)
             
             vbox.addLayout(hbox)
             gb.setLayout(vbox)
             
             gb.setLayout(vbox)
             
             # Add to main layout. Insert before the last item (status label?) if possible, or just add.
             # count() - 1 usually works.
             if hasattr(self, "verticalLayout"):
                  self.verticalLayout.addWidget(gb)
             else:
                  self.layout().addWidget(gb)
             
        except Exception as e:
             print(f"Traj UI Setup Error: {e}")

    def _on_load_topology(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Topology/Structure", "", "Structure Files (*.pdb *.psf *.gro *.cif);;All Files (*)"
        )
        if fname:
            try:
                # Load into PyMOL. Use basename as object name by default PyMOL behavior
                cmd.load(fname)
                self._populate_limit_combo() # Refresh object list
                # Try to select the new object if possible
                bn = os.path.basename(fname)
                obj_name = os.path.splitext(bn)[0]
                idx = self.structure_combo.findText(obj_name)
                if idx >= 0:
                     self.structure_combo.setCurrentIndex(idx)
                QtWidgets.QMessageBox.information(self, "Success", f"Loaded {bn}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load topology: {e}")

    def _on_load_trajectory(self):
        # Must have a structure selected
        obj = self.structure_combo.currentText()
        if not obj:
             QtWidgets.QMessageBox.warning(self, "Error", "No structure selected. Load a topology first.")
             return
             
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Trajectory", "", "Trajectory Files (*.dcd *.xtc *.trr *.xyz *.nc);;All Files (*)"
        )
        if fname:
            try:
                cmd.load_traj(fname, obj)
                QtWidgets.QMessageBox.information(self, "Success", f"Loaded trajectory into '{obj}'")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load trajectory: {e}")

    def _update_features_coords_from_pymol(self, features):
        """Update coordinates in features dict from current PyMOL state."""
        # This allows us to use the static parsed structure (Bio.PDB) for topology/elements
        # but fetch dynamic coordinates from the trajectory.
        try:
             # Fetch all atoms for the object in current state
             # We rely on an ordered match or ID match.
             # cmd.get_model returns atoms.
             # To be safe, we map by (chain, resi, name)
             model = cmd.get_model(self.loaded_object)
             
             # Build lookup
             coord_map = {}
             for a in model.atom:
                  key = (a.chain, int(a.resi), a.name)
                  coord_map[key] = a.coord
                  
             # Update features lists (lists of dicts)
             # keys: protein_atoms, ligand_atoms, etc.
             # But 'protein_atoms' dicts are references. If we update them, referenced lists update too?
             # Yes, features["protein_atoms"] contains dict objects.
             # We need to update 'coord' key in each atom_info dict.
             
             # We can iterate 'protein_atoms' and 'ligand_atoms'.
             all_atom_lists = [features["protein_atoms"], features["ligand_atoms"]]
             
             for atom_list in all_atom_lists:
                  for atom_info in atom_list:
                       # key must match PDB parsing.
                       # Bio.PDB chain.id is str. res.id[1] is int. atom.name is str.
                       k = (atom_info["chain"], atom_info["resi"], atom_info["name"])
                       if k in coord_map:
                            atom_info["coord"] = np.array(coord_map[k])
                            
             # Re-compute rings/centroids since coords have changed
             features["protein_rings"] = self._find_rings(features["protein_atoms"], ATOM_DEFS["protein_aromatic_rings"])
             features["ligand_rings"] = self._detect_ligand_rings(features)
             if features["ligand_atoms"]:
                  features["ligand_centroid"] = get_centroid([a["coord"] for a in features["ligand_atoms"]])
                  
        except Exception as e:
             print(f"Coord Update Error: {e}")

    def _precompute_atom_features(self, update_coords=False):
        # Optimization: cache the topology features if structure hasn't changed?
        # For now, simplistic approach.
        # But for trajectory, we want to Reuse topology and just Update Coords.
        
        # If we have cached features and just want coord update:
        if update_coords and hasattr(self, "_cached_features"):
             features = self._cached_features
             self._update_features_coords_from_pymol(features)
             return features

        features = defaultdict(list)
        lig_chain, lig_resi, lig_resn = self.ligand_info

        for atom in self.structure.get_atoms():
            # Prefer highest occupancy altlocs
            try:
                if getattr(atom, "is_disordered", False) and atom.is_disordered() and hasattr(atom, "child_dict"):
                    # pick highest occupancy child
                    children = list(atom.child_dict.values())
                    if children:
                        atom = max(children, key=lambda a: (a.get_occupancy() or 0.0))
            except Exception:
                pass
            res = atom.parent
            chain = res.parent
            resn = res.get_resname()
            resi = res.id[1]
            is_ligand = chain.id == lig_chain and resi == lig_resi and resn == lig_resn

            atom_info = {
                "atom": atom,
                "resn": resn,
                "resi": resi,
                "chain": chain.id,
                "coord": atom.coord,
                "name": atom.name,
                "element": atom.element,
                "origin": "ligand" if is_ligand else ("protein" if is_aa(res) else "other"),
            }

            if is_ligand:
                features["ligand_atoms"].append(atom_info)
                if atom.element in ATOM_DEFS["halogens"]:
                    features["ligand_halogens"].append(atom_info)
                if atom.element in ["O", "N"]:
                    features["ligand_h_donors_acceptors"].append(atom_info)
                if atom.element == "C":
                    features["ligand_hydrophobic"].append(atom_info)
                if atom.element in ["O", "N", "S"]:
                    features["ligand_metal_acceptors"].append(atom_info)
            elif is_aa(res):
                features["protein_atoms"].append(atom_info)
                if resn in ATOM_DEFS["protein_positive"] and atom.name in ATOM_DEFS["protein_positive"][resn]:
                    features["protein_positive"].append(atom_info)
                if resn in ATOM_DEFS["protein_negative"] and atom.name in ATOM_DEFS["protein_negative"][resn]:
                    features["protein_negative"].append(atom_info)
                if (resn in ATOM_DEFS["protein_h_donors"] and atom.name in ATOM_DEFS["protein_h_donors"][resn]) or (
                    atom.name == "N" and resn != "PRO"
                ):
                    features["protein_h_donors"].append(atom_info)
                if (
                    resn in ATOM_DEFS["protein_h_acceptors"] and atom.name in ATOM_DEFS["protein_h_acceptors"][resn]
                ) or (atom.name == "O"):
                    features["protein_h_acceptors"].append(atom_info)
                if resn in ATOM_DEFS["protein_hydrophobic_res"] and atom.element == "C":
                    features["protein_hydrophobic"].append(atom_info)
            elif resn in ATOM_DEFS["metals"]:
                features["metals"].append(atom_info)

        features["protein_rings"] = self._find_rings(features["protein_atoms"], ATOM_DEFS["protein_aromatic_rings"])
        features["ligand_rings"] = self._detect_ligand_rings(features)

        # Store ligand centroid for legend placement
        # Store ligand centroid for legend placement
        if features["ligand_atoms"]:
            # --- CHARGE CALCULATION LOGIC ---
            try:
                # 1. Reconstruct Mol from Ligand Atoms (PDB Block)
                # Sort by Serial number (if available) or order seen
                # We need to preserve order for MolFromPDBBlock to work? No, just valid PDB lines.
                pdb_lines = []
                for i, atom_info in enumerate(features["ligand_atoms"]):
                    # simple PDB ATOM format: "ATOM  %5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f           %2s  "
                    # Just need minimal columns
                    atom = atom_info["atom"]
                    c = atom_info["coord"]
                    
                    # Sanitize Element (Bio.PDB sometimes leaks atom names like 'CB')
                    el = atom_info['element'].strip().upper()
                    # If invalid or looks like atom name
                    if len(el) > 2 or el in ("CB", "CA", "CG", "CD", "CE", "CZ", "CH", "OE", "OD", "NE", "NZ"): 
                         # Guess from name if possible or default to C?
                         name = atom_info['name'].strip().upper()
                         # Simple heuristic
                         if name.startswith("CL"): el = "Cl"
                         elif name.startswith("BR"): el = "Br"
                         elif name.startswith("FE"): el = "Fe"
                         elif name.startswith("MG"): el = "Mg"
                         elif name.startswith("ZN"): el = "Zn"
                         elif name.startswith("MN"): el = "Mn"
                         elif len(name) > 0 and name[0] in "CNOSP":
                             el = name[0] 
                         else:
                             el = "C" # Desperate fallback
                             
                    # Explicit mapping for common Bio.PDB errors
                    if el == "CB": el = "C" # Shouldn't happen with above, but safety
                    
                    # Use a minimal generator or string format
                    s = f"ATOM  {i+1:>5d} {atom_info['name']:<4s} {atom_info['resn']:>3s} {atom_info['chain']:1s}{atom_info['resi']:>4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00           {el:>2s}  "
                    pdb_lines.append(s)
                
                pdb_block = "\n".join(pdb_lines)
                mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
                
                if mol:
                     charges = self._compute_ligand_charges(mol)
                     # Annotate atoms
                     for atom_info in features["ligand_atoms"]:
                          name = atom_info["name"]
                          if name in charges:
                               atom_info["charge_formal"] = charges[name]["formal"]
                               atom_info["charge_partial"] = charges[name]["partial"]
                          else:
                               atom_info["charge_formal"] = 0
                               atom_info["charge_partial"] = 0.0
            except Exception as e:
                print(f"Charge Integration Error: {e}")
            
            features["ligand_centroid"] = get_centroid([a["coord"] for a in features["ligand_atoms"]])
        else:
            features["ligand_centroid"] = None
            
        # Cache for trajectory reuse
        self._cached_features = features

        return features

    def _calc_sasa(self):
        try:
            if not self.ligand_info:
                return None
                
            lig_chain, lig_resi, lig_resn = self.ligand_info
            # Ensure the object exists using the loaded object name
            obj = self.loaded_object
            if not obj:
                return None
                
            lig_sel = f"{obj} and chain {lig_chain} and resi {lig_resi} and resn {lig_resn}"
            
            # Verify selection has atoms
            if cmd.count_atoms(lig_sel) == 0:
                print(f"SASA Error: Ligand selection '{lig_sel}' is empty.")
                return None
            
            # Save current dot settings to restore later
            old_dot_sol = cmd.get_setting_text("dot_solvent")
            old_dot_den = cmd.get_setting_text("dot_density")
            
            cmd.set("dot_solvent", 1)
            cmd.set("dot_density", 2)
            
            # 1. Complex SASA (Ligand bound to protein)
            # We need to calculate the area of the ligand *in the context of the protein*
            # get_area returns the area of the selection.
            # However, if we just select the ligand, get_area might ignore the protein shielding 
            # unless we tell it to consider the context.
            # state=1, load_b=1 implies storing values in b-factor.
            # A more robust way:
            #   get_area selection, state=state, load_b=0
            # Calculation of 'bound' state:
            #   The SASA of the ligand atoms when the protein is present.
            
            # Make a complex selection to ensure context is considered? 
            # Actually get_area behavior depends on 'selection'. It calculates area for those atoms.
            # But occlusion from other atoms is only considered if they are part of the object/enabled.
            # Assuming protein and ligand are in the same object '{obj}' and enabled.
            
            complex_sasa = cmd.get_area(lig_sel)
            
            # 2. Free Ligand SASA
            # We must isolate the ligand to calculate its fully exposed surface.
            # We can create a temporary object or just use 'disable protein' trick?
            # Safer to extract ligand to a temp object.
            temp_lig = "temp_lig_sasa_calc"
            cmd.create(temp_lig, lig_sel)
            free_sasa = cmd.get_area(temp_lig)
            cmd.delete(temp_lig)
            
            # Restore settings
            if old_dot_sol: cmd.set("dot_solvent", old_dot_sol)
            if old_dot_den: cmd.set("dot_density", old_dot_den)
            
            if free_sasa <= 0.001:
                return None # Avoid division by zero
                
            utils_buried_area = free_sasa - complex_sasa
            buried_perc = (utils_buried_area / free_sasa) * 100.0
            
            return {
                "bound": complex_sasa,
                "free": free_sasa,
                "buried_percent": buried_perc
            }

        except Exception as e:
            print(f"SASA Calculation failed: {e}")
            return None

    def _on_show_2d_map(self):
        if not HAS_RDKIT_MPL:
            QtWidgets.QMessageBox.warning(self, "Error", "RDKit and Matplotlib are required for 2D maps.")
            return
        if not self.ligand_info:
            return
        try:
            self._generate_2d_map()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to generate 2D map: {e}")

    def _generate_2d_map(self):
        lig_chain, lig_resi, lig_resn = self.ligand_info
        lig_sel = f"{self.loaded_object} and chain {lig_chain} and resi {lig_resi} and resn {lig_resn}"
        
        # Get PDB block for ligand
        pdb_block = cmd.get_pdbstr(lig_sel)
        if not pdb_block:
            raise ValueError("Could not retrieve ligand PDB block")
            
        mol = Chem.MolFromPDBBlock(pdb_block)
        if not mol:
            raise ValueError("RDKit failed to parse ligand PDB block")
            
        # Compute 2D coords (preserve 3D first)
        conf = mol.GetConformer()
        coords3d = [np.array(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        
        AllChem.Compute2DCoords(mol)
        conf2d = mol.GetConformer()
        
        # Map atom names to RDKit indices and coords
        atom_map = {} # name -> (idx, x, y)
        coords_2d = {} # idx -> np.array([x, y])
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf2d.GetAtomPosition(idx)
            p = np.array([pos.x, pos.y])
            coords_2d[idx] = p
            
            info = atom.GetPDBResidueInfo()
            if info:
                name = info.GetName().strip()
                atom_map[name] = {"idx": idx, "pos": p}
        
        # Pre-calc Ring Centroids in 3D for mapping
        ring_map = [] # (centroid_3d, atom_indices)
        ri = mol.GetRingInfo()
        if ri:
             for ring_idxs in ri.AtomRings():
                  # Compute 3D centroid
                  pts = [coords3d[i] for i in ring_idxs]
                  c_3d = np.mean(pts, axis=0)
                  ring_map.append((c_3d, ring_idxs))
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 9)) # Larger figure
        ax.set_aspect('equal')
        ax.set_axis_off()
        
        # Calculate Mol Center and Radius (for placement)
        all_p = np.array(list(coords_2d.values()))
        mol_center = np.mean(all_p, axis=0)
        mol_radius = np.max(np.linalg.norm(all_p - mol_center, axis=1))
        label_radius = mol_radius + 2.5 # Place labels this far out
        
        # --- Draw Ligand Bonds ---
        for bond in mol.GetBonds():
            b = bond.GetBeginAtomIdx()
            e = bond.GetEndAtomIdx()
            p1 = coords_2d[b]
            p2 = coords_2d[e]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=2, zorder=1)
            
        # --- Draw Ligand Atoms ---
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            p = coords_2d[idx]
            symbol = atom.GetSymbol()
            c = "black"
            if symbol == "O": c = "red"
            elif symbol == "N": c = "blue"
            elif symbol == "S": c = "orange"
            elif symbol == "P": c = "purple"
            elif symbol == "F" or symbol == "Cl": c = "green"
            
            if symbol != "C":
                ax.text(p[0], p[1], symbol, color=c, fontsize=10, 
                        ha='center', va='center', fontweight='bold', 
                        bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="none", alpha=0.8), zorder=2)

        # --- Process Interactions for Layout ---
        interactions_to_plot = []
        
        for k, inter in enumerate(self.interactions):
            try:
                l_name = inter.get('lig_atom')
                p_res = inter.get('prot_res')
                itype = inter.get('type')
                dist = inter.get('distance', 0)
                l_coord = inter.get('lig_coord') # 3D tuple
                
                # Resolve start position (Ligand Atom or Ring Center in 2D)
                start_pos = None
                
                # Method 1: Exact Name Match
                if l_name and l_name in atom_map:
                    start_pos = atom_map[l_name]["pos"]
                
                # Method 2: Coordinate Match (Robust)
                if start_pos is None and l_coord:
                     l_c_arr = np.array(l_coord)
                     
                     # Check atoms (threshold 0.5A)
                     best_atom_idx = -1
                     best_dist = 0.5
                     for i, c3 in enumerate(coords3d):
                          d = np.linalg.norm(c3 - l_c_arr)
                          if d < best_dist:
                               best_dist = d
                               best_atom_idx = i
                     
                     if best_atom_idx != -1:
                          start_pos = coords_2d[best_atom_idx]
                     
                     # Check Rings (if looking for ring or if single atom failed)
                     if start_pos is None and (l_name == "Ring" or "ligand_ring" in inter.get("extra", {})):
                          best_ring_idx = -1
                          best_r_dist = 1.0 # Centroid might slightly deviate
                          for i, (rc, r_idxs) in enumerate(ring_map):
                               d = np.linalg.norm(rc - l_c_arr)
                               if d < best_r_dist:
                                    best_r_dist = d
                                    best_ring_idx = i
                          
                          if best_ring_idx != -1:
                               # Calculate 2D centroid of this ring
                               r_idxs = ring_map[best_ring_idx][1]
                               pts_2d = [coords_2d[ri] for ri in r_idxs]
                               start_pos = np.mean(pts_2d, axis=0)
                
                # Method 3: Fallback to Mol Center
                if start_pos is None and l_name == "Ring":
                     start_pos = mol_center

                if start_pos is not None:
                    # Calculate angle from center
                    vec = start_pos - mol_center
                    angle = np.arctan2(vec[1], vec[0])
                    interactions_to_plot.append({
                        "angle": angle,
                        "start_pos": start_pos,
                        "itype": itype,
                        "label": f"{itype}\n{p_res}\n{dist:.2f}Å",
                        "color": COLOR_MAP.get(itype, 'gray')
                    })
            except Exception:
                pass
        
        # --- Overlap Avoidance (Angular Spreading) ---
        # Sort by angle
        interactions_to_plot.sort(key=lambda x: x["angle"])
        
        if interactions_to_plot:
            min_sep = 0.35 # radians (~20 degrees)
            
            # Simple iterative spreading
            # We iterate multiple times to smooth out distribution
            for _ in range(5):
                for i in range(len(interactions_to_plot)):
                    curr = interactions_to_plot[i]
                    # Check next
                    next_idx = (i + 1) % len(interactions_to_plot)
                    next_item = interactions_to_plot[next_idx]
                    
                    # Diff
                    diff = next_item["angle"] - curr["angle"]
                    if next_idx == 0: # Wrap around
                        diff = (next_item["angle"] + 2*np.pi) - curr["angle"]
                        
                    if diff < min_sep:
                        # Push apart
                        push = (min_sep - diff) / 2.0
                        curr["angle"] -= push
                        next_item["angle"] += push
                        
                        # Normalize angles to -pi, pi range isn't strictly needed for drawing but good for logic
                        # But here strictly wrapping logic matters. 
                        # Simplified: Just ensure spacing.
            
            # --- Draw Lines and Labels ---
            for item in interactions_to_plot:
                angle = item["angle"]
                start_pos = item["start_pos"]
                
                # Calculate label position
                # Label is placed at fixed radius from MOLECULE CENTER
                # This ensures they form a nice circle around the ligand
                lx = mol_center[0] + np.cos(angle) * label_radius
                ly = mol_center[1] + np.sin(angle) * label_radius
                label_pos = np.array([lx, ly])
                
                # Draw Line: Ligand Atom -> Label Pos
                ax.plot([start_pos[0], label_pos[0]], [start_pos[1], label_pos[1]], 
                        '--', color=item["color"], linewidth=1, alpha=0.6)
                
                # Draw Label
                # Alignment depends on side (left/right)
                ha = 'left' if lx >= mol_center[0] else 'right'
                
                ax.text(lx, ly, item["label"], color='black', fontsize=8,
                        ha=ha, va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=item["color"], alpha=0.9, lw=1.5))

        plt.title(f"Ligand Interaction Map ({lig_resn} {lig_resi})")
        plt.tight_layout()
        plt.show()

    def _run_trajectory_analysis(self):
        if not HAS_RDKIT_MPL:
             QtWidgets.QMessageBox.warning(self, "Error", "Matplotlib is required for plotting.")
             return
             
        try:
             states = cmd.count_states(self.loaded_object)
             if states < 2:
                  QtWidgets.QMessageBox.warning(self, "Warning", "Only 1 state detected. Load a trajectory first.")
                  return
                  
             step = self.traj_step.value()
             
             # Prepare
             print(f"Starting analysis on {states} frames with step {step}...")
             interaction_counts = defaultdict(int) 
             # key: (type, prot_res, lig_atom), value: count
             
             # Ensure topology is cached
             self._precompute_atom_features(update_coords=False)
             
             frames_analyzed = 0
             
             # Progress Bar (Dialog?)
             pd = QtWidgets.QProgressDialog("Analyzing Trajectory...", "Cancel", 0, states, self)
             pd.setWindowModality(QtCore.Qt.WindowModal)
             
             for i in range(1, states + 1, step):
                  if pd.wasCanceled():
                       break
                  pd.setValue(i)
                  
                  cmd.frame(i)
                  # Update method needs to be rigorous
                  # Since we use `_calculate_all_interactions` which calls `_precompute_atom_features`
                  # We need to tell it to USE UPDATE MODE.
                  # But `_calculate_all_interactions` doesn't take args in my current design.
                  # Modifications needed:
                  # 1. Update `_calculate_all_interactions` to accept `update_coords=True`?
                  # 2. Or just verify `_precompute_atom_features` logic handles it?
                  # My previous edit added `update_coords=False` default.
                  # So `_calculate_all_interactions` calls `self._precompute_atom_features()`.
                  # I need to change that call.
                  
                  # Hack: Set a flag on self?
                  self._traj_mode = True
                  current_inters = self._calculate_all_interactions()
                  self._traj_mode = False
                  
                  frames_analyzed += 1
                  for inter in current_inters:
                       # Respect User Filters (Type Checkboxes)
                       if hasattr(self, "_type_checkboxes"):
                            itype = inter["type"]
                            cb = self._type_checkboxes.get(itype)
                            # If checkbox exists and is unchecked, skip
                            if cb and not cb.isChecked():
                                 continue
                                 
                       # Key for uniqueness: Type + ProtRes + LigAtom
                       # Remove "distance" specific keys
                       k = (inter["type"], inter["prot_res"], inter["lig_atom"])
                       interaction_counts[k] += 1
                       
             pd.setValue(states)
             
             if not interaction_counts:
                  QtWidgets.QMessageBox.information(self, "Result", "No interactions found in trajectory.")
                  return
                  
             # Plotting
             # Sort by frequency
             sorted_inters = sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True)
             top_n = min(len(sorted_inters), 20)
             top_items = sorted_inters[:top_n]
             
             labels = [f"{k[1]}-{k[2]}" for k, v in top_items]
             freqs = [(v/frames_analyzed)*100 for k, v in top_items] # Percentage
             colors = [COLOR_MAP.get(k[0], "gray") for k, v in top_items]
             
             plt.figure(figsize=(10, 6))
             bars = plt.bar(range(top_n), freqs, color=colors)
             plt.xticks(range(top_n), labels, rotation=45, ha='right')
             plt.ylabel("Occupancy (%)")
             plt.title(f"Trajectory Interaction Analysis ({frames_analyzed} frames)")
             plt.tight_layout()
             
             # Legend logic for colors?
             # Or just color by type. Labels are specific residue pairs.
             # Maybe add type label?
             
             plt.show()
             
        except Exception as e:
             QtWidgets.QMessageBox.critical(self, "Error", f"Trajectory Analysis failed: {e}")
             print(f"Traceback: {e}")

    class _NeighborGrid:
        def __init__(self, points, coords_key="coord", cell_size=4.0):
            self.cell = float(cell_size)
            self.coords_key = coords_key
            self.grid = defaultdict(list)
            for obj in points:
                x, y, z = obj[coords_key]
                key = (int(x // self.cell), int(y // self.cell), int(z // self.cell))
                self.grid[key].append(obj)

        def query(self, center, radius):
            cx, cy, cz = center
            r = float(radius)
            rx = range(int((cx - r) // self.cell), int((cx + r) // self.cell) + 1)
            ry = range(int((cy - r) // self.cell), int((cy + r) // self.cell) + 1)
            rz = range(int((cz - r) // self.cell), int((cz + r) // self.cell) + 1)
            out = []
            for ix in rx:
                for iy in ry:
                    for iz in rz:
                        out.extend(self.grid.get((ix, iy, iz), []))
            return out

    def _bond_threshold(self, e1, e2):
        r1 = COVALENT_RADII.get(e1.upper(), 0.77)
        r2 = COVALENT_RADII.get(e2.upper(), 0.77)
        return r1 + r2 + BONDTOL

    def _neighbors(self, target, pool, element_filter=None):
        res = []
        for a in pool:
            if a is target:
                continue
            if (a["chain"], a["resi"]) != (target["chain"], target["resi"]):
                continue
            if element_filter and a["element"].upper() not in element_filter:
                continue
            d = distance(a["coord"], target["coord"])
            if d <= self._bond_threshold(a["element"], target["element"]):
                res.append(a)
        return res

    def _donor_hydrogens(self, donor, features):
        pool = features["ligand_atoms"] if donor["origin"] == "ligand" else features["protein_atoms"]
        return [n for n in self._neighbors(donor, pool) if n["element"].upper() == "H"]

    def _acceptor_heavy_neighbor(self, acceptor, features):
        pool = features["ligand_atoms"] if acceptor["origin"] == "ligand" else features["protein_atoms"]
        heavies = [n for n in self._neighbors(acceptor, pool) if n["element"].upper() != "H"]
        if not heavies:
            return None
        # choose closest heavy neighbor
        return min(heavies, key=lambda n: distance(n["coord"], acceptor["coord"]))

    def _find_rings(self, atoms, ring_defs):
        rings = []
        residues = defaultdict(list)
        for atom in atoms:
            residues[(atom["chain"], atom["resi"])].append(atom)

        for (chain, resi), res_atoms in residues.items():
            resn = res_atoms[0]["resn"]
            if resn in ring_defs:
                ring_atom_names = ring_defs[resn]
                ring_coords = [a["coord"] for name in ring_atom_names for a in res_atoms if a["name"] == name]
                if len(ring_coords) == len(ring_atom_names):
                    centroid = get_centroid(ring_coords)
                    normal = get_normal(ring_coords)
                    radius = get_ring_radius(ring_coords, centroid)
                    rings.append(
                        {
                            "resn": resn,
                            "resi": resi,
                            "chain": chain,
                            "coords": ring_coords,
                            "centroid": centroid,
                            "normal": normal,
                            "radius": radius,
                        }
                    )
            elif not ring_defs:
                pass
        return rings

    def _compute_ligand_charges(self, mol):
        """
        Compute partial and formal charges for the ligand.
        If hydrogens are missing, use Dimorphite-DL (if available) to determine
        protonation state at pH 7.4, then transfer formal charges.
        Then compute Gasteiger charges.
        
        Returns: dict { atom_name: {'formal': int, 'partial': float} }
        """
        try:
            # Check for Hydrogens
            has_h = False
            for a in mol.GetAtoms():
                if a.GetAtomicNum() == 1:
                    has_h = True
                    break
            
            target_mol = mol
            
            # If no hydrogens, attempt PH-dependent protonation to get Formal Charges
            if not has_h:
                try:
                    import dimorphite_dl
                    from rdkit.Chem import Descriptors
                    
                    smiles = Chem.MolToSmiles(mol)
                    # Use specified pH or default 7.4
                    target_ph = getattr(self, "_target_ph", None)
                    if target_ph is None:
                         target_ph = 7.4
                    
                    protonated_smiles_list = dimorphite_dl.protonate(smiles, min_ph=target_ph-0.2, max_ph=target_ph+0.2)
                    
                    if protonated_smiles_list:
                         # Take the first dominant state
                         p_smi = protonated_smiles_list[0]
                         p_mol = Chem.MolFromSmiles(p_smi)
                         
                         if p_mol:
                             # Transfer Formal Charges to original Mol based on graph match
                             # Matches heavy atoms
                             matches = mol.GetSubstructMatches(p_mol, useChirality=True)
                             if not matches:
                                  matches = mol.GetSubstructMatches(p_mol, useChirality=False) # relaxed
                             
                             if matches:
                                  # match is a tuple of indices in 'mol' corresponding to 'p_mol' atoms
                                  # But p_mol atoms are ordered by SMILES.
                                  # We need to transfer charge FROM p_mol TO mol.
                                  match = matches[0] # Take first match
                                  for i, mol_idx in enumerate(match):
                                       p_atom = p_mol.GetAtomWithIdx(i)
                                       mol_atom = mol.GetAtomWithIdx(mol_idx)
                                       mol_atom.SetFormalCharge(p_atom.GetFormalCharge())
                                       
                             target_mol = mol # Update in place
                except ImportError:
                    pass
                except Exception as e:
                    print(f"Protonation Error: {e}")

            # Compute Gasteiger Charges
            # (Requires Hydrogens for accuracy? If explicit H missing, results might be off unless ImplicitValence used)
            # RDKit Gasteiger usually needs H.
            # If we didn't add explicit H above (we just set formal charge), we might want to AddHs for calculation?
            # But we can't AddHs if we don't change coords?
            # We can work on a COPY for calculation.
            
            calc_mol = Chem.AddHs(target_mol) 
            AllChem.ComputeGasteigerCharges(calc_mol)
            
            # Map back to names
            charges = {}
            for atom in calc_mol.GetAtoms():
                 # We only care about Heavy Atoms existing in original mol
                 # AddHs keeps original indices logic? No.
                 # But we can match by Name? RDKit doesn't persist PDB Names easily unless sanitized properly.
                 # Actually, if we loaded from PDB Block, 'mol' has PDBInfo.
                 # 'calc_mol' (AddHs) should preserve Info?
                 info = atom.GetPDBResidueInfo()
                 if info:
                      name = info.GetName().strip()
                      fullname = info.GetName()
                      # Gasteiger
                      gc = float(atom.GetDoubleProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
                      fc = atom.GetFormalCharge()
                      charges[name] = {"formal": fc, "partial": gc}
            
            return charges
            
        except Exception as e:
            print(f"Charge Calc Error: {e}")
            return {}
            
    def _detect_ligand_rings(self, features):
        # Build adjacency for ligand atoms
        lig = features["ligand_atoms"]
        if not lig:
            return []
        # Map index for stability
        idx_map = {id(a): i for i, a in enumerate(lig)}
        coords = [a["coord"] for a in lig]
        elements = [a["element"].upper() for a in lig]
        adj = {i: set() for i in range(len(lig))}
        for i in range(len(lig)):
            for j in range(i + 1, len(lig)):
                if distance(coords[i], coords[j]) <= self._bond_threshold(elements[i], elements[j]):
                    adj[i].add(j)
                    adj[j].add(i)

        # Find simple cycles of length 5 or 6 via DFS, canonicalize by sorted tuple
        cycles = set()
        target_lengths = {5, 6}

        def dfs(start, current, visited):
            if len(current) > 6:
                return
            u = current[-1]
            for v in adj[u]:
                if v == start and len(current) in target_lengths:
                    cyc = tuple(sorted(current))
                    cycles.add(cyc)
                if v in visited or v < start:  # simple ordering to reduce duplicates
                    continue
                dfs(start, current + [v], visited | {v})

        for s in range(len(lig)):
            dfs(s, [s], {s})

        rings = []
        lig_chain, lig_resi, lig_resn = self.ligand_info
        for cyc in cycles:
            ring_coords = [coords[i] for i in cyc]
            # planarity check
            normal = get_normal(ring_coords)
            centroid = get_centroid(ring_coords)
            # distance of each atom from plane
            plane_d = []
            for p in ring_coords:
                plane_d.append(abs(np.dot(normal, p - centroid)))
            if max(plane_d) > 0.2:  # Å
                continue
            # aromatic-like composition
            en = [elements[i] for i in cyc]
            aro_frac = sum(1 for e in en if e in ("C", "N")) / len(en)
            if aro_frac < 0.8:
                continue
            radius = get_ring_radius(ring_coords, centroid)
            rings.append(
                {
                    "resn": lig_resn,
                    "resi": lig_resi,
                    "chain": lig_chain,
                    "coords": ring_coords,
                    "centroid": centroid,
                    "normal": normal,
                    "radius": radius,
                }
            )
        return rings

    def _calculate_all_interactions(self):
        """Main calculation function that orchestrates finding all interaction types."""
        if not self.structure or not self.ligand_info:
            return []

        # Coordinate Update Logic for Trajectory
        update_coords = getattr(self, "_traj_mode", False)
        features = self._precompute_atom_features(update_coords=update_coords)
        interactions = []

        # Helper to append interactions de-duplicated and with consistent formatting
        seen = set()

        def add_interaction(itype, prot, lig, dist, details=None, extra=None):
            prot_res = f"{prot.get('resn', '?')} {prot.get('chain', '?')}{prot.get('resi', '?')}"
            prot_atom = prot.get("name", "")
            lig_atom = lig.get("name", "")
            key = (itype, prot_res, prot_atom, lig_atom)
            if key in seen:
                return
            seen.add(key)
            rec = {
                "type": itype,
                "prot_res": prot_res,
                "prot_atom": prot_atom,
                "lig_atom": lig_atom,
                "distance": float(dist),
                "details": details or "",
            }
            
            # Store coordinates (essential for 2D map mapping if names fail)
            # Try direct 'coord' first, then ring 'centroid' from extra
            l_c = lig.get("coord")
            if l_c is None and extra and isinstance(extra, dict) and "ligand_ring" in extra:
                 l_c = extra["ligand_ring"].get("centroid")
            if l_c is not None:
                 rec["lig_coord"] = tuple(l_c) # Store as tuple for safety
                 
            p_c = prot.get("coord") 
            if p_c is None and extra and isinstance(extra, dict) and "prot_ring" in extra:
                 p_c = extra["prot_ring"].get("centroid")
            if p_c is not None:
                 rec["prot_coord"] = tuple(p_c)

            if isinstance(extra, dict):
                rec.update(extra)
            interactions.append(rec)

        # --- Hydrogen bonds with angle checks (using neighbor grids) ---
        hb_cut = GEOMETRY_CRITERIA["h_bond_dist"]
        grid_lig_acc = self._NeighborGrid(features["ligand_h_donors_acceptors"], cell_size=hb_cut)
        grid_prot_acc = self._NeighborGrid(features["protein_h_acceptors"], cell_size=hb_cut)

        def is_hbond(donor, acceptor):
            dDA = distance(donor["coord"], acceptor["coord"])
            if dDA > GEOMETRY_CRITERIA["h_bond_dist"]:
                return False, dDA, None
            # Prefer D-H-A angle if H present
            Hs = self._donor_hydrogens(donor, features)
            if Hs:
                best = max(
                    angle_between_vectors(h["coord"] - donor["coord"], acceptor["coord"] - h["coord"]) for h in Hs
                )
                return (best >= GEOMETRY_CRITERIA["h_bond_angle"]), dDA, f"DHA: {best:.1f}°"
            # Fallback: D-A-X angle at acceptor using nearest heavy neighbor X
            a_nb = self._acceptor_heavy_neighbor(acceptor, features)
            if a_nb:
                ang = angle_between_vectors(donor["coord"] - acceptor["coord"], a_nb["coord"] - acceptor["coord"])
                return (ang >= GEOMETRY_CRITERIA["h_bond_angle"]), dDA, f"DAX: {ang:.1f}°"
            return True, dDA, None  # as last resort accept distance-only

        # Protein donor -> Ligand acceptor
        # --- Hydrogen Bonds (Strict D-H...A) ---
        # User Requirement: "must get hydrogen in the middle"
        # We iterate Donors, find attached Hydrogens, and check geometry with Acceptors.
        
        hb_dist_HA = 2.8  # Strict H...A distance max (Angstroms)
        hb_angle_min = 90.0 # Min D-H...A angle (degrees) - usually >120 preferred but 90 allows for some flexibility
        
        # Protein Donor (Heavy) -> Ligand Acceptor
        for p_d in features["protein_h_donors"]:
            # Find attached hydrogens in protein
            p_Hs = self._donor_hydrogens(p_d, features)
            if not p_Hs: continue
            
            for h in p_Hs:
                # Check interaction with Ligand Acceptors near H
                for l_a in grid_lig_acc.query(h["coord"], hb_dist_HA):
                    if l_a["element"] not in ("O", "N", "F", "S"): continue
                    
                    # Geometry Check
                    d_HA = distance(h["coord"], l_a["coord"])
                    if d_HA > hb_dist_HA: continue
                    
                    # D-H...A Angle
                    # vector DH = H - D, vector HA = A - H. Angle is between DH and HA? No, typically D-H..A angle.
                    # Vector H->D: p_d - h. Vector H->A: l_a - h.
                    # Angle is typically defined at H? No, angle D-H...A is usually ~180.
                    # Let's use `angle_between_vectors` which returns angle between 0-180.
                    v_HD = p_d["coord"] - h["coord"]  # Vector H to D
                    v_HA = l_a["coord"] - h["coord"]  # Vector H to A
                    # Wait, angle_between_vectors(v1, v2). If linear D-H...A, v_HD and v_HA are opposite?
                    # D -- H ... A
                    # H-D vector points left. H-A vector points right. Angle 180.
                    
                    angle = angle_between_vectors(v_HD, v_HA)
                    if angle < hb_angle_min: continue
                    
                    # Store Interaction (use Heavy atom distance for consistent visualization/labeling usually, but H is scientifically correct)
                    # Standard Pymol distance is often Heavy-Heavy. But let's use Heavy-Heavy for the "dashed line" anchor?
                    # Or should we anchor to H? User said "hydrogen in the middle".
                    # Let's interact D...A but note H involvement.
                    d_DA = distance(p_d["coord"], l_a["coord"])
                    add_interaction("Hydrogen Bond", p_d, l_a, d_DA, f"H-Bond (H...A {d_HA:.1f}Å, {angle:.0f}°)")

        # Ligand Donor (Heavy) -> Protein Acceptor
        for l_d in features["ligand_h_donors_acceptors"]:
            if l_d["element"] not in ("O", "N", "S"): continue # Donors are usually N/O/S
            # Find attached hydrogens in ligand
            l_Hs = self._donor_hydrogens(l_d, features)
            if not l_Hs: continue
            
            for h in l_Hs:
                # Check interaction with Protein Acceptors near H
                for p_a in grid_prot_acc.query(h["coord"], hb_dist_HA):
                    if p_a["element"] not in ("O", "N", "S"): continue

                    d_HA = distance(h["coord"], p_a["coord"])
                    if d_HA > hb_dist_HA: continue
                    
                    v_HD = l_d["coord"] - h["coord"]
                    v_HA = p_a["coord"] - h["coord"]
                    angle = angle_between_vectors(v_HD, v_HA)
                    
                    if angle < hb_angle_min: continue
                    
                    d_DA = distance(l_d["coord"], p_a["coord"])
                    add_interaction("Hydrogen Bond", p_a, l_d, d_DA, f"H-Bond (H...A {d_HA:.1f}Å, {angle:.0f}°)")

        # --- Salt bridges (One per residue pair) ---
        # Collect all candidates first, then filter.
        sb_candidates = []
        sb_cut = GEOMETRY_CRITERIA["salt_bridge_dist"]
        # Ligand Negative atoms (Strict Charge-Based)
        # Element + Charge check (Formal <= -1 or Partial <= -0.3)
        lig_neg_atoms = []
        for a in features["ligand_atoms"]:
             if a["element"] in ("O", "S", "P", "F", "Cl", "Br", "I"):
                  q_f = a.get("charge_formal", 0)
                  q_p = a.get("charge_partial", 0.0)
                  if q_f <= -1 or q_p <= -0.3:
                       lig_neg_atoms.append(a)
        
        # Ligand Positive atoms (Cation-Pi)
        # Element N + Charge check (Formal >= 1 or Partial >= 0.3)
        lig_pos_atoms = []
        for a in features["ligand_atoms"]:
             if a["element"] == "N" and a["name"] != "N":
                  q_f = a.get("charge_formal", 0)
                  q_p = a.get("charge_partial", 0.0)
                  if q_f >= 1 or q_p >= 0.3:
                       lig_pos_atoms.append(a)
        
        # Salt Bridge Candidates (Protein Negative -> Ligand Positive, Protein Positive -> Ligand Negative)
        
        # Ligand Negative candidates
        grid_lig_neg = self._NeighborGrid(lig_neg_atoms, cell_size=sb_cut)
        for p_pos in features["protein_positive"]:
            for l_neg in grid_lig_neg.query(p_pos["coord"], sb_cut):
                d = distance(p_pos["coord"], l_neg["coord"])
                if d <= sb_cut:
                    sb_candidates.append(("Salt Bridge", p_pos, l_neg, d))
        
        # Ligand Positive candidates
        grid_lig_pos = self._NeighborGrid(lig_pos_atoms, cell_size=sb_cut)
        for p_neg in features["protein_negative"]:
            for l_pos in grid_lig_pos.query(p_neg["coord"], sb_cut):
                d = distance(p_neg["coord"], l_pos["coord"])
                if d <= sb_cut:
                     sb_candidates.append(("Salt Bridge", p_neg, l_pos, d))

        # Filter: Group by (Protein Chain, Resi) -> Keep min distance one
        sb_by_res = {}
        for item in sb_candidates:
            itype, prot, lig, d = item
            key = (prot["chain"], prot["resi"]) # Ligand is always the same residue in this context
            if key not in sb_by_res or d < sb_by_res[key][3]:
                sb_by_res[key] = item
        
        for item in sb_by_res.values():
            add_interaction(*item)

        # --- Metal coordination ---
        mc_cut = GEOMETRY_CRITERIA["metal_coordination_dist"]
        grid_lig_macc = self._NeighborGrid(features["ligand_metal_acceptors"], cell_size=mc_cut)
        for metal in features["metals"]:
            for l_acc in grid_lig_macc.query(metal["coord"], mc_cut):
                d = distance(metal["coord"], l_acc["coord"])
                if d <= mc_cut:
                    add_interaction("Metal Coordination", metal, l_acc, d)

        # --- Halogen bonds ---
        hb_hal_cut = GEOMETRY_CRITERIA["halogen_dist"]
        grid_prot_acc2 = self._NeighborGrid(features["protein_h_acceptors"], cell_size=hb_hal_cut)
        for l_hal in features["ligand_halogens"]:
            neighbors = self._neighbors(l_hal, features["ligand_atoms"], element_filter={"C"})
            carbon = neighbors[0] if neighbors else None
            for p_acc in grid_prot_acc2.query(l_hal["coord"], hb_hal_cut):
                d = distance(l_hal["coord"], p_acc["coord"])
                if d <= GEOMETRY_CRITERIA["halogen_dist"]:
                    detail = None
                    if carbon is not None:
                         ang = angle_between_vectors(carbon["coord"] - l_hal["coord"], p_acc["coord"] - l_hal["coord"])
                         if ang < GEOMETRY_CRITERIA["halogen_angle"]: continue
                         detail = f"CXA: {ang:.1f}°"
                    add_interaction("Halogen Bond", p_acc, l_hal, d, detail)

        # --- Pi-System interactions ---
        # Cation-pi: cationic protein atom to ligand ring (if any) and vice versa
        cp_cut = GEOMETRY_CRITERIA["cation_pi_dist"]
        grid_lig_posN = self._NeighborGrid(lig_pos_atoms, cell_size=cp_cut)
        
        for p_ring in features["protein_rings"]:
            # Placeholder needs explicit "Ring" name for logic later
            p_placeholder = {"resn": p_ring["resn"], "resi": p_ring["resi"], "chain": p_ring["chain"], "name": "Ring"}
            for l_pos in grid_lig_posN.query(p_ring["centroid"], cp_cut):
                d = distance(l_pos["coord"], p_ring["centroid"])
                if d <= cp_cut:
                    add_interaction("Cation-Pi", p_placeholder, l_pos, d, extra={"prot_ring": p_ring})

        for l_ring in features["ligand_rings"]:
            l_placeholder = {"resn": l_ring["resn"], "resi": l_ring["resi"], "chain": l_ring["chain"], "name": "Ring"}
            grid_p_pos = self._NeighborGrid(features["protein_positive"], cell_size=cp_cut)
            for p_pos in grid_p_pos.query(l_ring["centroid"], cp_cut):
                d = distance(p_pos["coord"], l_ring["centroid"])
                angle = angle_between_vectors(l_ring["normal"], p_pos["coord"] - l_ring["centroid"])
                if d <= GEOMETRY_CRITERIA["cation_pi_dist"] and angle <= GEOMETRY_CRITERIA["cation_pi_angle"]:
                    add_interaction(
                        "Cation-Pi", p_pos, l_placeholder, d, f"Angle: {angle:.1f}°", extra={"ligand_ring": l_ring}
                    )
        
        # Anion-Pi: anionic protein atom to ligand ring and vice versa
        # Reuse 'lig_neg_atoms' defined for Salt Bridges if available, else redefine
        # (It is available from above scope)
        grid_lig_neg = self._NeighborGrid(lig_neg_atoms, cell_size=cp_cut)
        for p_ring in features["protein_rings"]:
            p_placeholder = {"resn": p_ring["resn"], "resi": p_ring["resi"], "chain": p_ring["chain"], "name": "Ring"}
            for l_neg in grid_lig_neg.query(p_ring["centroid"], cp_cut):
                 d = distance(l_neg["coord"], p_ring["centroid"])
                 if d <= cp_cut:
                      add_interaction("Anion-Pi", p_placeholder, l_neg, d, extra={"prot_ring": p_ring})
                      
        for l_ring in features["ligand_rings"]:
            l_placeholder = {"resn": l_ring["resn"], "resi": l_ring["resi"], "chain": l_ring["chain"], "name": "Ring"}
            grid_p_neg = self._NeighborGrid(features["protein_negative"], cell_size=cp_cut)
            for p_neg in grid_p_neg.query(l_ring["centroid"], cp_cut):
                 d = distance(p_neg["coord"], l_ring["centroid"])
                 # Angle check optional for Anion-Pi, but usually geometry is similar (normal vs charge vector)
                 angle = angle_between_vectors(l_ring["normal"], p_neg["coord"] - l_ring["centroid"])
                 if d <= GEOMETRY_CRITERIA["cation_pi_dist"]: # Reuse dist criteria
                      # Check angle? 
                      if angle <= GEOMETRY_CRITERIA["cation_pi_angle"]:
                           add_interaction("Anion-Pi", p_neg, l_placeholder, d, f"Angle: {angle:.1f}°", extra={"ligand_ring": l_ring})
            # ring-ring
            grid_p_rings = self._NeighborGrid(features["protein_rings"], cell_size=GEOMETRY_CRITERIA["pi_t_dist"], coords_key="centroid")
            for p_ring in grid_p_rings.query(l_ring["centroid"], GEOMETRY_CRITERIA["pi_t_dist"]):
                p_placeholder = {
                    "resn": p_ring["resn"],
                    "resi": p_ring["resi"],
                    "chain": p_ring["chain"],
                    "name": "Ring",
                }
                d = distance(p_ring["centroid"], l_ring["centroid"])
                angle = angle_between_vectors(p_ring["normal"], l_ring["normal"])
                if d <= GEOMETRY_CRITERIA["pi_pi_dist"] and (
                    angle <= GEOMETRY_CRITERIA["pi_pi_angle"] or angle >= 180 - GEOMETRY_CRITERIA["pi_pi_angle"]
                ):
                    add_interaction(
                        "Pi-Pi Stacking", p_placeholder, l_placeholder, d, f"Angle: {angle:.1f}°",
                        extra={"prot_ring": p_ring, "ligand_ring": l_ring},
                    )
                elif (
                    GEOMETRY_CRITERIA["pi_pi_dist"] < d <= GEOMETRY_CRITERIA["pi_t_dist"]
                    and GEOMETRY_CRITERIA["pi_t_angle_low"] <= angle <= GEOMETRY_CRITERIA["pi_t_angle_high"]
                ):
                    add_interaction(
                        "T-Shaped Pi-Pi", p_placeholder, l_placeholder, d, f"Angle: {angle:.1f}°",
                        extra={"prot_ring": p_ring, "ligand_ring": l_ring},
                    )

        # --- Cluster-based interactions (Hydrophobic and VdW) with neighbor grid ---
        hydrophobic_contacts = defaultdict(lambda: {"prot_atoms": set(), "lig_atoms": set(), "min_dist": 1e9})
        vdw_contacts = defaultdict(lambda: {"prot_atoms": set(), "lig_atoms": set(), "min_dist": 1e9})

        max_r = max(GEOMETRY_CRITERIA["hydrophobic_dist"], GEOMETRY_CRITERIA["vdw_dist"])
        p_grid = self._NeighborGrid(features["protein_atoms"], cell_size=max_r)

        for l_atom in features["ligand_atoms"]:
            neighbors = p_grid.query(l_atom["coord"], max_r)

            for p_atom in neighbors:
                d = distance(p_atom["coord"], l_atom["coord"])
                if d > max_r:
                    continue
                is_hydrophobic = (
                    p_atom["resn"] in ATOM_DEFS["protein_hydrophobic_res"]
                    and p_atom["element"] == "C"
                    and l_atom["element"] == "C"
                    and d <= GEOMETRY_CRITERIA["hydrophobic_dist"]
                )
                prot_key = (p_atom["chain"], p_atom["resi"])
                if is_hydrophobic:
                    hydrophobic_contacts[prot_key]["prot_atoms"].add(p_atom["name"])
                    hydrophobic_contacts[prot_key]["lig_atoms"].add(l_atom["name"])
                    hydrophobic_contacts[prot_key]["min_dist"] = min(hydrophobic_contacts[prot_key]["min_dist"], d)
                elif d <= GEOMETRY_CRITERIA["vdw_dist"]:
                    vdw_contacts[prot_key]["prot_atoms"].add(p_atom["name"])
                    vdw_contacts[prot_key]["lig_atoms"].add(l_atom["name"])
                    vdw_contacts[prot_key]["min_dist"] = min(vdw_contacts[prot_key]["min_dist"], d)

        # Materialize cluster interactions
        for (chain, resi), data in hydrophobic_contacts.items():
            if not data["prot_atoms"]:
                continue
            # infer residue name
            try:
                resn = next(a["resn"] for a in features["protein_atoms"] if a["chain"] == chain and a["resi"] == resi)
            except StopIteration:
                continue
            interactions.append(
                {
                    "type": "Hydrophobic",
                    "prot_res": f"{resn} {chain}{resi}",
                    "prot_atom": ",".join(sorted(data["prot_atoms"])),
                    "lig_atom": ",".join(sorted(data["lig_atoms"])),
                    "distance": float(data["min_dist"]),
                    "details": "Cluster",
                }
            )

        for (chain, resi), data in vdw_contacts.items():
            if not data["prot_atoms"]:
                continue
            try:
                resn = next(a["resn"] for a in features["protein_atoms"] if a["chain"] == chain and a["resi"] == resi)
            except StopIteration:
                continue
            # Skip if this residue already has a more specific interaction
            if any(
                i for i in interactions if i["prot_res"] == f"{resn} {chain}{resi}" and i["type"] != "Van der Waals"
            ):
                continue
            interactions.append(
                {
                    "type": "Van der Waals",
                    "prot_res": f"{resn} {chain}{resi}",
                    "prot_atom": ",".join(sorted(data["prot_atoms"])),
                    "lig_atom": ",".join(sorted(data["lig_atoms"])),
                    "distance": float(data["min_dist"]),
                    "details": "Cluster",
                }
            )

        return sorted(interactions, key=lambda x: (x["type"], x["distance"]))
        found_pairs = set()

        def add_interaction(inter_type, prot_atom, lig_atom, dist, details=""):
            prot_key = (prot_atom["chain"], prot_atom["resi"], prot_atom["name"])
            lig_key = (self.ligand_info[0], self.ligand_info[1], lig_atom["name"])
            pair_key = tuple(sorted((prot_key, lig_key)))
            if pair_key in found_pairs:
                return
            found_pairs.add(pair_key)
            interactions.append(
                {
                    "type": inter_type,
                    "prot_res": f"{prot_atom['resn']} {prot_atom['chain']}{prot_atom['resi']}",
                    "prot_atom": prot_atom["name"],
                    "lig_atom": lig_atom["name"],
                    "distance": dist,
                    "details": details,
                }
            )

        # --- Specific Interactions (atom-to-atom) ---
        for p_don in features["protein_h_donors"]:
            for l_acc in features["ligand_h_donors_acceptors"]:
                dist = distance(p_don["coord"], l_acc["coord"])
                if dist <= GEOMETRY_CRITERIA["h_bond_dist"]:
                    add_interaction("Hydrogen Bond", p_don, l_acc, dist)
        for l_don in features["ligand_h_donors_acceptors"]:
            for p_acc in features["protein_h_acceptors"]:
                dist = distance(l_don["coord"], p_acc["coord"])
                if dist <= GEOMETRY_CRITERIA["h_bond_dist"]:
                    add_interaction("Hydrogen Bond", p_acc, l_don, dist)

        for p_pos in features["protein_positive"]:
            for l_neg in features["ligand_atoms"]:
                if l_neg["element"] in ["O", "N"]:
                    dist = distance(p_pos["coord"], l_neg["coord"])
                    if dist <= GEOMETRY_CRITERIA["salt_bridge_dist"]:
                        add_interaction("Salt Bridge", p_pos, l_neg, dist)
        for p_neg in features["protein_negative"]:
            for l_pos in features["ligand_atoms"]:
                if l_pos["element"] == "N":
                    dist = distance(p_neg["coord"], l_pos["coord"])
                    if dist <= GEOMETRY_CRITERIA["salt_bridge_dist"]:
                        add_interaction("Salt Bridge", p_neg, l_pos, dist)

        for metal in features["metals"]:
            for l_acc in features["ligand_metal_acceptors"]:
                dist = distance(metal["coord"], l_acc["coord"])
                if dist <= GEOMETRY_CRITERIA["metal_coordination_dist"]:
                    add_interaction("Metal Coordination", metal, l_acc, dist)

        for l_hal in features["ligand_halogens"]:
            for p_acc in features["protein_h_acceptors"]:
                dist = distance(l_hal["coord"], p_acc["coord"])
                if dist <= GEOMETRY_CRITERIA["halogen_dist"]:
                    add_interaction("Halogen Bond", p_acc, l_hal, dist)

        # --- Pi-System Interactions ---
        for p_ring in features["protein_rings"]:
            p_placeholder = {"resn": p_ring["resn"], "resi": p_ring["resi"], "chain": p_ring["chain"], "name": "Ring"}
            for l_pos in features["ligand_atoms"]:
                if l_pos["element"] == "N":
                    dist = distance(l_pos["coord"], p_ring["centroid"])
                    if dist <= GEOMETRY_CRITERIA["cation_pi_dist"]:
                        add_interaction("Cation-Pi", p_placeholder, l_pos, dist)

        for l_ring in features["ligand_rings"]:
            l_placeholder = {"resn": l_ring["resn"], "resi": l_ring["resi"], "chain": l_ring["chain"], "name": "Ring"}
            for p_pos in features["protein_positive"]:
                dist = distance(p_pos["coord"], l_ring["centroid"])
                angle = angle_between_vectors(l_ring["normal"], p_pos["coord"] - l_ring["centroid"])
                if dist <= GEOMETRY_CRITERIA["cation_pi_dist"] and angle <= GEOMETRY_CRITERIA["cation_pi_angle"]:
                    add_interaction("Cation-Pi", p_pos, l_placeholder, dist, f"Angle: {angle:.1f}°")
            for p_ring in features["protein_rings"]:
                p_placeholder = {
                    "resn": p_ring["resn"],
                    "resi": p_ring["resi"],
                    "chain": p_ring["chain"],
                    "name": "Ring",
                }
                dist = distance(p_ring["centroid"], l_ring["centroid"])
                angle = angle_between_vectors(p_ring["normal"], l_ring["normal"])
                if dist <= GEOMETRY_CRITERIA["pi_pi_dist"] and (
                    angle <= GEOMETRY_CRITERIA["pi_pi_angle"] or angle >= 180 - GEOMETRY_CRITERIA["pi_pi_angle"]
                ):
                    add_interaction("Pi-Pi Stacking", p_placeholder, l_placeholder, dist, f"Angle: {angle:.1f}°")
                elif (
                    GEOMETRY_CRITERIA["pi_pi_dist"] < dist <= GEOMETRY_CRITERIA["pi_t_dist"]
                    and GEOMETRY_CRITERIA["pi_t_angle_low"] <= angle <= GEOMETRY_CRITERIA["pi_t_angle_high"]
                ):
                    add_interaction("T-Shaped Pi-Pi", p_placeholder, l_placeholder, dist, f"Angle: {angle:.1f}°")

        # --- Cluster-based Interactions (Hydrophobic and VdW) ---
        hydrophobic_contacts = defaultdict(lambda: {"prot_atoms": set(), "lig_atoms": set(), "min_dist": 100.0})
        vdw_contacts = defaultdict(lambda: {"prot_atoms": set(), "lig_atoms": set(), "min_dist": 100.0})

        for p_atom in features["protein_atoms"]:
            for l_atom in features["ligand_atoms"]:
                dist = distance(p_atom["coord"], l_atom["coord"])
                is_hydrophobic_contact = (
                    p_atom["resn"] in ATOM_DEFS["protein_hydrophobic_res"]
                    and p_atom["element"] == "C"
                    and l_atom["element"] == "C"
                    and dist <= GEOMETRY_CRITERIA["hydrophobic_dist"]
                )

                prot_key = (p_atom["chain"], p_atom["resi"])
                if is_hydrophobic_contact:
                    hydrophobic_contacts[prot_key]["prot_atoms"].add(p_atom["name"])
                    hydrophobic_contacts[prot_key]["lig_atoms"].add(l_atom["name"])
                    if dist < hydrophobic_contacts[prot_key]["min_dist"]:
                        hydrophobic_contacts[prot_key]["min_dist"] = dist
                elif dist <= GEOMETRY_CRITERIA["vdw_dist"]:
                    vdw_contacts[prot_key]["prot_atoms"].add(p_atom["name"])
                    vdw_contacts[prot_key]["lig_atoms"].add(l_atom["name"])
                    if dist < vdw_contacts[prot_key]["min_dist"]:
                        vdw_contacts[prot_key]["min_dist"] = dist

        for (chain, resi), data in hydrophobic_contacts.items():
            resn = next(a["resn"] for a in features["protein_atoms"] if a["chain"] == chain and a["resi"] == resi)
            interactions.append(
                {
                    "type": "Hydrophobic",
                    "prot_res": f"{resn} {chain}{resi}",
                    "prot_atom": ",".join(sorted(data["prot_atoms"])),
                    "lig_atom": ",".join(sorted(data["lig_atoms"])),
                    "distance": data["min_dist"],
                    "details": "Cluster",
                }
            )

        for (chain, resi), data in vdw_contacts.items():
            resn = next(a["resn"] for a in features["protein_atoms"] if a["chain"] == chain and a["resi"] == resi)
            is_specific = any(
                i for i in interactions if i["prot_res"] == f"{resn} {chain}{resi}" and i["type"] != "Hydrophobic"
            )
            if not is_specific:
                interactions.append(
                    {
                        "type": "Van der Waals",
                        "prot_res": f"{resn} {chain}{resi}",
                        "prot_atom": ",".join(sorted(data["prot_atoms"])),
                        "lig_atom": ",".join(sorted(data["lig_atoms"])),
                        "distance": data["min_dist"],
                        "details": "Cluster",
                    }
                )

        return sorted(interactions, key=lambda x: (x["type"], x["distance"]))

    def _populate_table(self):
        self.table.setRowCount(0)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Type", "Prot Res", "Prot Atom", "Lig Atom", "Distance (Å)", "Details"])
        for i, inter in enumerate(self.interactions):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(inter["type"]))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(inter["prot_res"]))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(inter["prot_atom"]))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(inter["lig_atom"]))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{inter['distance']:.2f}"))
            self.table.setItem(i, 5, QtWidgets.QTableWidgetItem(inter.get("details", "")))
        self.table.resizeColumnsToContents()

    def _on_table_clicked(self, row, col):
        try:
            inter = self._table_items[row] if self._table_items else self.interactions[row]
            rid = self.interactions.index(inter)
        except Exception:
            return
        self._toggle_interaction_by_id(rid)

    def _toggle_interaction_by_id(self, rid):
        inter = self.interactions[rid]
        res_name = f"intRes_{rid}"
        dash_name = f"int_dash_{rid}"
        cloud_prot_name = f"intCloudProt_{rid}"
        cloud_lig_name = f"intCloudLig_{rid}"
        disc_prot_name = f"intRingDiscProt_{rid}"
        disc_lig_name = f"intRingDiscLig_{rid}"

        if rid in self._selected_ids:
            try:
                cmd.delete(res_name)
                cmd.delete(dash_name)
                cmd.delete(cloud_prot_name)
                cmd.delete(cloud_lig_name)
                cmd.delete(disc_prot_name)
                cmd.delete(disc_lig_name)
                # Cleanup labels
                cmd.delete(f"lbl_prot_{rid}")
                cmd.delete(f"lbl_lig_{rid}")
                cmd.delete(f"lbl_dist_{rid}")
            except Exception:
                pass
            self._selected_ids.remove(rid)
            return
        else:
            prot_chain = inter["prot_res"].split()[1][0]
            prot_resi = inter["prot_res"].split()[1][1:]
            prot_atom_names = inter["prot_atom"]
            lig_chain, lig_resi, lig_resn = self.ligand_info
            lig_atom_names = inter["lig_atom"]
            prot_chain = inter["prot_res"].split()[1][0]
            prot_resi = inter["prot_res"].split()[1][1:]
            prot_atom_names = inter["prot_atom"]
            lig_chain, lig_resi, lig_resn = self.ligand_info
            lig_atom_names = inter["lig_atom"]

            # Show the interacting protein residue
            prot_res_sel = f"{self.loaded_object} and chain {prot_chain} and resi {prot_resi}"
            inter_type_group = f"Interactions.{inter['type'].replace(' ', '_')}"
            try:
                cmd.select(res_name, prot_res_sel)
                cmd.show("sticks", res_name)
                cmd.show("spheres", res_name)
                cmd.set("sphere_scale", 0.25, res_name)
                cmd.set("stick_radius", 0.15, res_name)
                # Coloring: Light Orange Carbon for contrast against white cartoon
                # util.cbao colors carbons Light Orange, others by element
                cmd.util.cbao(res_name) 
                
                cmd.group(inter_type_group, res_name)
            except Exception:
                pass

            # Visualization logic
            if inter.get("details") == "Cluster":
                # --- Cloud Visualization for Clusters ---
                prot_atom_list = prot_atom_names.split(",")
                lig_atom_list = lig_atom_names.split(",")
                prot_cloud_sel = f"{prot_res_sel} and name " + "+".join(prot_atom_list)
                lig_cloud_sel = f"{self.loaded_object} and chain {lig_chain} and resi {lig_resi} and name " + "+".join(
                    lig_atom_list
                )

                try:
                    cmd.select(cloud_prot_name, prot_cloud_sel)
                    cmd.select(cloud_lig_name, lig_cloud_sel)
                    cmd.show("surface", cloud_prot_name)
                    cmd.show("surface", cloud_lig_name)
                    # per-type cloud opacity
                    cov = (
                        self._settings.get("style_overrides", {}).get(inter["type"], {}).get("cloud_opacity")
                        if hasattr(self, "_settings")
                        else None
                    )
                    trans = 1.0 - float(cov) if cov is not None else 0.6
                    cmd.set("transparency", trans, cloud_prot_name)
                    cmd.set("transparency", trans, cloud_lig_name)
                    ctype = self._type_color_name(inter["type"])
                    cmd.color(ctype, cloud_prot_name)
                    cmd.color(ctype, cloud_lig_name)
                    cmd.group(inter_type_group, cloud_prot_name)
                    cmd.group(inter_type_group, cloud_lig_name)
                except Exception:
                    pass
            else:
                # Visualization logic
                # Use pseudoatoms for Ring centers to allow 'cmd.distance' to work properly
                
                p1_sel = None
                p2_sel = None
            
                # PROTEIN SIDE
                if inter["prot_atom"] == "Ring" or "prot_coord" in inter:
                     # It's a ring or special coordinate. Create pseudoatom.
                     p1_coord = inter.get("prot_coord")
                     if p1_coord:
                          p1_name = f"pseudo_p_{rid}"
                          cmd.pseudoatom(p1_name, pos=list(p1_coord))
                          p1_sel = p1_name
                     else:
                          pass
                else:
                     p1_sel = f"({prot_res_sel} and name {prot_atom_names})"

                # LIGAND SIDE
                if inter["lig_atom"] == "Ring" or "lig_coord" in inter:
                     p2_coord = inter.get("lig_coord")
                     if p2_coord:
                          p2_name = f"pseudo_l_{rid}"
                          cmd.pseudoatom(p2_name, pos=list(p2_coord))
                          p2_sel = p2_name
                else:
                     p2_sel = f"({self.loaded_object} and chain {lig_chain} and resi {lig_resi} and name {lig_atom_names})"

                if p1_sel and p2_sel:
                     try:
                          # Create distance object (Primary Visualization)
                          cmd.distance(dash_name, p1_sel, p2_sel)
                          
                          # Check if valid distance created
                          # Apply Styles
                          c_name = f"c_{inter['type'].replace(' ', '_')}"
                          c_rgb = _hex_to_rgb(COLOR_MAP.get(inter["type"], "#FFFFFF"))
                          if c_rgb:
                              cmd.set_color(c_name, list(c_rgb))
                              cmd.set("dash_color", c_name, dash_name)
                          
                          # Style from map
                          style = STYLE_MAP.get(inter["type"], {})
                          radius = style.get("dash_radius", 0.05)
                          cmd.set("dash_radius", radius, dash_name)
                          cmd.set("dash_gap", style.get("dash_gap", 0.2), dash_name)
                          cmd.set("dash_length", style.get("dash_length", 0.3), dash_name)
                          cmd.set("dash_round_ends", 1, dash_name)
                          
                          # Ensure label is visible? cmd.distance shows label by default.
                          cmd.set("label_color", c_name, dash_name)
                          cmd.set("label_size", 20, dash_name) 
                          
                     except Exception as e:
                          print(f"Viz Error {rid}: {e}")
                
                cmd.group(inter_type_group, dash_name)
                    
                # --- Interactive Dynamic Labels (Restored) ---
                # Labels: Bold, Green, No Outline
                
                lbl_prot_name = f"lbl_prot_{rid}"
                lbl_lig_name = f"lbl_lig_{rid}"
                
                # Label Protein Atom: "Resn Resi Atom" e.g. "ASN 123 OD1"
                # Need p1 coordinate if it was a Ring
                if inter["prot_atom"] == "Ring" and "prot_coord" in inter:
                     p1_c = inter["prot_coord"]
                     cmd.pseudoatom(lbl_prot_name, pos=list(p1_c))
                     cmd.label(lbl_prot_name, '"%s %s Ring"' % (inter["prot_res"].split()[0], inter["prot_res"].split()[1]))
                else:
                     prot_atom_sel = f"({prot_res_sel} and name {prot_atom_names})"
                     cmd.select(lbl_prot_name, prot_atom_sel)
                     cmd.label(lbl_prot_name, '"%s %s %s" % (resn, resi, name)')

                cmd.set("label_color", "green", lbl_prot_name)
                # cmd.set("label_outline_color", "black", lbl_prot_name) # Removed borders per request
                cmd.set("label_font_id", 2, lbl_prot_name) # 2 = Sans Bold
                
                # Label Ligand Atom: "Atom" e.g. "N1"
                if inter["lig_atom"] == "Ring" and "lig_coord" in inter:
                     p2_c = inter["lig_coord"]
                     cmd.pseudoatom(lbl_lig_name, pos=list(p2_c))
                     cmd.label(lbl_lig_name, '"Ring"')
                else:
                     # Re-derive selector if needed, or assume it's valid
                     lig_atom_sel = f"({self.loaded_object} and chain {lig_chain} and resi {lig_resi} and name {lig_atom_names})"
                     cmd.select(lbl_lig_name, lig_atom_sel) 
                     cmd.label(lbl_lig_name, '"%s" % (name)') 

                cmd.set("label_color", "green", lbl_lig_name)
                # cmd.set("label_outline_color", "black", lbl_lig_name) # Removed borders per request
                cmd.set("label_font_id", 2, lbl_lig_name) # 2 = Sans Bold
                
                cmd.group(inter_type_group, lbl_prot_name)
                cmd.group(inter_type_group, lbl_lig_name)
                
                # Distance Label is handled by cmd.distance object itself (dash_name)
                # So we don't need manual pseudoatom for it regarding 'p1'/'p2'
                pass



                    # Optional angle label - REMOVED per user request for cleaner look
                    # if bool(self._settings.get("show_angle_labels", False)) ...


            # --- Ring plane discs for pi interactions ---
            try:
                ring_color = inter["type"]
                if "prot_ring" in inter:
                    r = inter["prot_ring"]
                    self._make_ring_disc(disc_prot_name, r["centroid"], r["normal"], r.get("radius", 1.8), ring_color)
                    cmd.group(inter_type_group, disc_prot_name)
                if "ligand_ring" in inter:
                    r = inter["ligand_ring"]
                    self._make_ring_disc(disc_lig_name, r["centroid"], r["normal"], r.get("radius", 1.8), ring_color)
                    cmd.group(inter_type_group, disc_lig_name)
            except Exception:
                pass

            self._selected_ids.add(rid)

    def _beautify_scene(self):
        """Sets up a nice, clean visualization scene."""
        if not self.loaded_object:
            return
        cmd.hide("everything", "all")

        # Set a nice dark grey background
        cmd.bg_color("grey20")

        # Better lighting
        cmd.set("ray_trace_mode", 1)
        cmd.set("specular", 0.4)
        cmd.set("shininess", 50)
        cmd.set("ambient", 0.3)

        # Show polymer (protein/nucleic acids) as clean white cartoon, excluding selected ligand residue
        prot_sel = f"{self.loaded_object} and polymer"
        if self.ligand_info:
            c, i, r = self.ligand_info
            lig_excl = f" and not (chain {c} and resi {i} and resn {r})"
            prot_sel += lig_excl
        cmd.show("cartoon", prot_sel)
        cmd.set("cartoon_color", "white", prot_sel)

        # Show ligand in licorice/ball-and-stick, colored by element
        if self.ligand_info:
            c, i, r = self.ligand_info
            lig_sel = f"{self.loaded_object} and chain {c} and resi {i} and resn {r}"
            try:
                cmd.show("sticks", lig_sel)
                cmd.set("stick_radius", 0.25, lig_sel)
                cmd.util.cba(20, lig_sel)  # Color by element
                if bool(self._settings.get("auto_zoom", True)) if hasattr(self, "_settings") else True:
                    cmd.zoom(lig_sel, 6)
            except Exception:
                pass

        # General settings for a clean look
        cmd.set("antialias", 2)
        # Dash settings serve as backup if CGO fails
        cmd.set("dash_gap", 0.0) 
        cmd.set("dash_radius", 0.1)
        cmd.set("ray_trace_fog", 0)  # No fog
        cmd.set("depth_cue", 0)  # No depth cueing
        # Legend overlay refresh
        try:
            self._update_legend()
        except Exception:
            pass

    def _apply_dash_style(self, obj_name, itype):
        style = STYLE_MAP.get(itype, {}).copy()
        # apply overrides
        try:
            o = self._settings.get("style_overrides", {}).get(itype, {}) if hasattr(self, "_settings") else {}
            style.update({k: v for k, v in o.items() if k in ("dash_radius", "dash_length", "dash_gap")})
        except Exception:
            pass
        if not style:
            return
        try:
            if "dash_radius" in style:
                cmd.set("dash_radius", style["dash_radius"], obj_name)
            if "dash_length" in style:
                cmd.set("dash_length", style["dash_length"], obj_name)
            if "dash_gap" in style:
                cmd.set("dash_gap", style["dash_gap"], obj_name)
        except Exception:
            pass

    def _type_color_name(self, itype):
        # Resolve color name for type considering overrides
        base = COLOR_MAP.get(itype, "yellow")
        try:
            o = self._settings.get("style_overrides", {}).get(itype, {}) if hasattr(self, "_settings") else {}
            hexc = o.get("color")
            if hexc:
                rgb = _hex_to_rgb(hexc)
                if rgb:
                    cname = f"pli_color_{itype.replace(' ','_')}"
                    if _COLOR_CACHE.get(cname) != rgb:
                        cmd.set_color(cname, list(rgb))
                        _COLOR_CACHE[cname] = rgb
                    return cname
        except Exception:
            pass
        return base

    def _make_ring_disc(self, name, centroid, normal, radius, color_or_type):
        thickness = 0.10
        # per-type override thickness
        if isinstance(color_or_type, str) and color_or_type in COLOR_MAP:
            try:
                ov = (
                    self._settings.get("style_overrides", {}).get(color_or_type, {})
                    if hasattr(self, "_settings")
                    else {}
                )
                tval = ov.get("disc_thickness")
                if tval is not None:
                    thickness = float(tval)
            except Exception:
                pass
        # fallback to global control
        if hasattr(self, "disc_thickness_spin"):
            try:
                if thickness == 0.10:  # only if not overridden
                    thickness = float(self.disc_thickness_spin.value())
            except Exception:
                pass
        c = np.array(centroid)
        n = np.array(normal)
        if np.linalg.norm(n) == 0:
            n = np.array([0.0, 0.0, 1.0])
        n = n / np.linalg.norm(n)
        p1 = (c - n * (thickness / 2.0)).tolist()
        p2 = (c + n * (thickness / 2.0)).tolist()
        # Resolve color from type overrides or color name
        if isinstance(color_or_type, str) and color_or_type in COLOR_MAP:
            cname = self._type_color_name(color_or_type)
            rgb = cmd.get_color_tuple(cname)
        else:
            cname = color_or_type if isinstance(color_or_type, str) else None
            rgb = cmd.get_color_tuple(cname) if cname else color_or_type
        if not rgb:
            rgb = (1.0, 1.0, 0.0)
        obj = [
            cgo.CYLINDER,
            p1[0],
            p1[1],
            p1[2],
            p2[0],
            p2[1],
            p2[2],
            float(radius),
            rgb[0],
            rgb[1],
            rgb[2],
            rgb[0],
            rgb[1],
            rgb[2],
        ]
        cmd.load_cgo(obj, name)
        try:
            # per-type disc opacity
            if isinstance(color_or_type, str) and color_or_type in COLOR_MAP:
                ov = (
                    self._settings.get("style_overrides", {}).get(color_or_type, {})
                    if hasattr(self, "_settings")
                    else {}
                )
                op = ov.get("disc_opacity")
                trans = 1.0 - float(op) if op is not None else 0.5
            else:
                trans = 0.5
            cmd.set("transparency", trans, name)
        except Exception:
            pass

    def _on_render_export(self):
        out = self.png_path_edit.text().strip() or "output.png"
        cmd.set("ray_trace_mode", 1)
        # place export-safe legend
        try:
            include = True
            if hasattr(self, "legend_export_cb"):
                include = self.legend_export_cb.isChecked()
            if include:
                self._update_legend(for_export=True, anchor=self._legend_anchor())
            else:
                cmd.delete("Interactions.Legend")
            # Scale bar and title
            if bool(self._settings.get("include_scale", True)) if hasattr(self, "_settings") else True:
                self._update_scale_bar_for_export()
            else:
                cmd.delete("Interactions.ScaleBar")
            if bool(self._settings.get("include_title", False)) if hasattr(self, "_settings") else False:
                self._update_title_for_export()
            else:
                cmd.delete("Interactions.Title")
        except Exception:
            pass
        cmd.png(out, width=2000, height=2000, dpi=300, ray=1)
        QtWidgets.QMessageBox.information(self, "Exported", f"Image saved to {out}")

    def _update_scale_bar_for_export(self):
        # remove previous
        try:
            cmd.delete("Interactions.ScaleBar")
        except Exception:
            pass
        try:
            L = float(self._settings.get("scale_length", 10)) if hasattr(self, "_settings") else 10.0
        except Exception:
            L = 10.0
        view = cmd.get_view()
        try:
            minv, maxv = cmd.get_extent(self.loaded_object)
            minv = np.array(minv)
            maxv = np.array(maxv)
            center = (minv + maxv) / 2.0
            diag = float(np.linalg.norm(maxv - minv)) or 10.0
        except Exception:
            center = np.array([view[12], view[13], view[14]])
            diag = 10.0
        right = np.array(view[0:3])
        up = np.array(view[3:6])
        right = right / (np.linalg.norm(right) or 1.0)
        up = up / (np.linalg.norm(up) or 1.0)
        # place near bottom-left by default; reuse legend anchor
        anchor = self._legend_anchor()
        frac = 0.35
        off_r = frac * diag
        off_u = frac * diag
        if anchor == "top_left":
            off_r = -off_r
        if anchor in ("bottom_left", "bottom_right"):
            off_u = -off_u
        base = np.array(center) + right * off_r + up * (off_u - 0.15 * diag)
        p1 = base
        p2 = base + right * L
        rgb = cmd.get_color_tuple("white") or (1.0, 1.0, 1.0)
        obj = [
            cgo.CYLINDER,
            float(p1[0]),
            float(p1[1]),
            float(p1[2]),
            float(p2[0]),
            float(p2[1]),
            float(p2[2]),
            0.1,
            rgb[0],
            rgb[1],
            rgb[2],
            rgb[0],
            rgb[1],
            rgb[2],
        ]
        cmd.load_cgo(obj, "Interactions.ScaleBar")
        # label
        try:
            lab = "scale_label"
            pos = p2 + up * (0.08 * diag)
            cmd.pseudoatom(lab, pos=[float(pos[0]), float(pos[1]), float(pos[2])])
            cmd.label(lab, f'"{int(round(L))} Å"')
            cmd.set("label_color", "white", lab)
            cmd.set("label_outline_color", "black", lab)
            cmd.group("Interactions.ScaleBar", lab)
        except Exception:
            pass

    def _update_title_for_export(self):
        try:
            cmd.delete("Interactions.Title")
        except Exception:
            pass
        text = ""
        try:
            text = str(self._settings.get("title_text", "")) if hasattr(self, "_settings") else ""
        except Exception:
            text = ""
        if not text:
            return
        view = cmd.get_view()
        try:
            minv, maxv = cmd.get_extent(self.loaded_object)
            minv = np.array(minv)
            maxv = np.array(maxv)
            center = (minv + maxv) / 2.0
            diag = float(np.linalg.norm(maxv - minv)) or 10.0
        except Exception:
            center = np.array([view[12], view[13], view[14]])
            diag = 10.0
        up = np.array(view[3:6])
        up = up / (np.linalg.norm(up) or 1.0)
        pos = center + up * (0.45 * diag)
        name = "title_label"
        cmd.pseudoatom(name, pos=[float(pos[0]), float(pos[1]), float(pos[2])])
        cmd.label(name, f'"{text}"')
        cmd.set("label_color", "white", name)
        cmd.set("label_outline_color", "black", name)
        cmd.group("Interactions.Title", name)

    def _on_show_hist(self):
        # Simple histogram dialog for distances
        try:
            from PySide2.QtWidgets import QDialog, QPlainTextEdit, QPushButton, QVBoxLayout
        except Exception:
            QDialog = QtWidgets.QDialog
            QVBoxLayout = QtWidgets.QVBoxLayout
            QPlainTextEdit = QtWidgets.QPlainTextEdit
            QPushButton = QtWidgets.QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("Distance Histogram")
        layout = QVBoxLayout(dlg)
        txt = QPlainTextEdit(dlg)
        txt.setReadOnly(True)
        layout.addWidget(txt)
        close_btn = QPushButton("Close", dlg)
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        # build bins 0-2,...,18-20
        bins = [(i, i + 2) for i in range(0, 20, 2)]
        counts = [0] * len(bins)
        dists = [i.get("distance", 0.0) for i in self._current_filtered_items()]
        for d in dists:
            for idx, (a, b) in enumerate(bins):
                if a <= d < b or (b == 20 and d <= 20):
                    counts[idx] += 1
                    break
        lines = []
        for (a, b), c in zip(bins, counts):
            bar = "▇" * min(c, 50)
            lines.append(f"{a:2d}-{b:2d}: {c:4d} {bar}")
        txt.setPlainText("\n".join(lines))
        dlg.resize(400, 300)
        dlg.exec_()

    def _on_edit_styles(self):
        # Dialog to edit per-type color and dash styles
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Edit Styles")
        layout = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(dlg)
        types = sorted(COLOR_MAP.keys())
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels(
            [
                "Type",
                "Color",
                "#Dash Radius",
                "#Dash Length",
                "#Dash Gap",
                "Disc Thickness",
                "Disc Opacity",
                "Cloud Opacity",
            ]
        )
        table.setRowCount(len(types))
        overrides = dict(self._settings.get("style_overrides", {})) if hasattr(self, "_settings") else {}

        def make_color_btn(color_hex):
            btn = QtWidgets.QPushButton()
            btn.setText(color_hex or "")
            if color_hex:
                btn.setStyleSheet(f"background:{color_hex};")

            def choose():
                col = QtWidgets.QColorDialog.getColor()
                if col.isValid():
                    hx = col.name()
                    btn.setText(hx)
                    btn.setStyleSheet(f"background:{hx};")

            btn.clicked.connect(choose)
            return btn

        for row, t in enumerate(types):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(t))
            table.item(row, 0).setFlags(table.item(row, 0).flags() & ~QtCore.Qt.ItemIsEditable)
            ov = overrides.get(t, {})
            color_btn = make_color_btn(ov.get("color"))
            table.setCellWidget(row, 1, color_btn)
            for col, key, default in [
                (2, "dash_radius", STYLE_MAP.get(t, {}).get("dash_radius", 0.08)),
                (3, "dash_length", STYLE_MAP.get(t, {}).get("dash_length", 0.4)),
                (4, "dash_gap", STYLE_MAP.get(t, {}).get("dash_gap", 0.2)),
            ]:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(3)
                spin.setRange(0.01, 2.0)
                spin.setSingleStep(0.01)
                spin.setValue(float(ov.get(key, default)))
                table.setCellWidget(row, col, spin)
            # Disc thickness per type (Å)
            thick_spin = QtWidgets.QDoubleSpinBox()
            thick_spin.setDecimals(2)
            thick_spin.setRange(0.02, 0.50)
            thick_spin.setSingleStep(0.01)
            default_thick = float(
                ov.get(
                    "disc_thickness", self.disc_thickness_spin.value() if hasattr(self, "disc_thickness_spin") else 0.10
                )
            )
            thick_spin.setValue(default_thick)
            table.setCellWidget(row, 5, thick_spin)
            # Disc opacity (0-1)
            disc_spin = QtWidgets.QDoubleSpinBox()
            disc_spin.setDecimals(2)
            disc_spin.setRange(0.0, 1.0)
            disc_spin.setSingleStep(0.05)
            disc_spin.setValue(float(ov.get("disc_opacity", 0.5)))
            table.setCellWidget(row, 6, disc_spin)
            # Cloud opacity (0-1)
            cloud_spin = QtWidgets.QDoubleSpinBox()
            cloud_spin.setDecimals(2)
            cloud_spin.setRange(0.0, 1.0)
            cloud_spin.setSingleStep(0.05)
            cloud_spin.setValue(float(ov.get("cloud_opacity", 0.4)))
            table.setCellWidget(row, 7, cloud_spin)
        layout.addWidget(table)
        btns = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        def on_save():
            new_ov = {}
            for row, t in enumerate(types):
                color_w = table.cellWidget(row, 1)
                dr = table.cellWidget(row, 2).value()
                dl = table.cellWidget(row, 3).value()
                dg = table.cellWidget(row, 4).value()
                thick = table.cellWidget(row, 5).value()
                disc = table.cellWidget(row, 6).value()
                cloud = table.cellWidget(row, 7).value()
                rec = {
                    "dash_radius": dr,
                    "dash_length": dl,
                    "dash_gap": dg,
                    "disc_thickness": thick,
                    "disc_opacity": disc,
                    "cloud_opacity": cloud,
                }
                color_hex = color_w.text().strip()
                if color_hex:
                    rec["color"] = color_hex
                new_ov[t] = rec
            self._settings["style_overrides"] = new_ov
            self._save_settings()
            # Apply to currently visible dashes by reapplying style
            try:
                for rid in list(self._selected_ids):
                    obj = f"int_dash_{rid}"
                    if cmd.count_atoms(obj) >= 0:
                        self._apply_dash_style(obj, self.interactions[rid]["type"])
                # Update legend colors
                self._update_legend(for_export=False, anchor=self._legend_anchor())
            except Exception:
                pass
            dlg.accept()

        save_btn.clicked.connect(on_save)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.resize(700, 400)
        dlg.exec_()

    # --- Enhancements: legend, clear, csv export, filtering ---
    def showEvent(self, event):
        super().showEvent(event)
        # Build legend dynamically once widgets exist
        try:
            self._load_settings()
            self._build_legend()
            self._init_filter()
            self._init_type_toggles()
            self._init_settings()
            self._apply_settings_to_ui()
            self._restore_last_session()
        except Exception:
            pass

    def _build_legend(self):
        # Legend removed per user request
        # If UI element exists, clear it
        try:
            legend_container = getattr(self, "legend_grid", None) or self.findChild(QtWidgets.QGridLayout, "legend_grid")
            if legend_container:
                while legend_container.count():
                    item = legend_container.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
        except:
            pass

    def _init_filter(self):
        combo = getattr(self, "filter_combo", None)
        if not combo:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All")
        for t in sorted(set(COLOR_MAP.keys())):
            combo.addItem(t)
        combo.blockSignals(False)
        combo.currentTextChanged.connect(lambda _t: (self._apply_filter(_t), self._save_settings()))
        # Max distance control
        if hasattr(self, "max_distance_spin"):
            try:
                self.max_distance_spin.setValue(0.00)
                self.max_distance_spin.valueChanged.connect(lambda _v: (self._populate_table(), self._save_settings()))
            except Exception:
                pass
        # Sync slider and spin
        if hasattr(self, "max_distance_slider") and hasattr(self, "max_distance_spin"):

            def slider_to_spin(val):
                try:
                    self.max_distance_spin.blockSignals(True)
                    self.max_distance_spin.setValue(val / 100.0)
                    self.max_distance_spin.blockSignals(False)
                    self._populate_table()
                    self._save_settings()
                except Exception:
                    pass

            def spin_to_slider(val):
                try:
                    self.max_distance_slider.blockSignals(True)
                    self.max_distance_slider.setValue(int(round(val * 100)))
                    self.max_distance_slider.blockSignals(False)
                except Exception:
                    pass

            self.max_distance_slider.valueChanged.connect(slider_to_spin)
            self.max_distance_spin.valueChanged.connect(spin_to_slider)

    def _init_type_toggles(self):
        layout = getattr(self, "types_layout", None) or self.findChild(QtWidgets.QVBoxLayout, "types_layout")
        if not layout:
            return
        # Clear any existing
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._type_checkboxes = {}
        types = sorted(COLOR_MAP.keys())
        saved = {}
        try:
            saved = self._settings.get("types_enabled", {}) if hasattr(self, "_settings") else {}
        except Exception:
            saved = {}
        for t in types:
            cb = QtWidgets.QCheckBox(t)
            checked = bool(saved.get(t, True))
            cb.setChecked(checked)
            cb.toggled.connect(lambda checked, tt=t: (self._on_type_toggle(tt, checked), self._save_settings()))
            layout.addWidget(cb)
            self._type_checkboxes[t] = cb
        spacer = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        layout.addItem(spacer)

    def _on_type_toggle(self, itype, checked):
        group = f"Interactions.{itype.replace(' ', '_')}"
        try:
            if checked:
                cmd.enable(group)
            else:
                cmd.disable(group)
        except Exception:
            pass

    def _apply_filter(self, text):
        # Rebuild table using current combined filters
        self._populate_table()

    def _init_settings(self):
        # Legend corner options
        if hasattr(self, "legend_corner_combo"):
            combo = self.legend_corner_combo
            if combo.count() == 0:
                combo.addItems(["Top Right", "Top Left", "Bottom Right", "Bottom Left"])
            combo.setCurrentText("Top Right")
            combo.currentTextChanged.connect(
                lambda _: (self._update_legend(for_export=False, anchor=self._legend_anchor()), self._save_settings())
            )
        # Legend export
        if hasattr(self, "legend_export_cb"):
            self.legend_export_cb.setChecked(True)
            self.legend_export_cb.toggled.connect(lambda _v: self._save_settings())
        # Disc thickness
        if hasattr(self, "disc_thickness_spin"):
            self.disc_thickness_spin.setValue(0.10)
            self.disc_thickness_spin.valueChanged.connect(
                lambda _v: (self._refresh_ring_discs(), self._save_settings())
            )
        # Distance labels
        if hasattr(self, "show_labels_cb"):
            self.show_labels_cb.setChecked(True)
            self.show_labels_cb.toggled.connect(lambda _v: (self._apply_labels_visibility(), self._save_settings()))
        # Compute on load
        if hasattr(self, "compute_on_load_cb"):
            self.compute_on_load_cb.setChecked(False)
            self.compute_on_load_cb.toggled.connect(lambda _v: self._save_settings())
        # Auto-zoom
        if hasattr(self, "auto_zoom_cb"):
            self.auto_zoom_cb.setChecked(True)
            self.auto_zoom_cb.toggled.connect(lambda _v: self._save_settings())
        # Max distance slider init
        if hasattr(self, "max_distance_slider") and hasattr(self, "max_distance_spin"):
            try:
                self.max_distance_slider.setValue(int(round(self.max_distance_spin.value() * 100)))
            except Exception:
                pass
        # Angle labels
        if hasattr(self, "show_angle_labels_cb"):
            self.show_angle_labels_cb.setChecked(False)
            self.show_angle_labels_cb.toggled.connect(lambda _v: self._save_settings())
        # Scale bar & title
        if hasattr(self, "include_scale_cb"):
            self.include_scale_cb.setChecked(True)
            self.include_scale_cb.toggled.connect(lambda _v: self._save_settings())
        if hasattr(self, "scale_length_spin"):
            self.scale_length_spin.setValue(10)
            self.scale_length_spin.valueChanged.connect(lambda _v: self._save_settings())
        if hasattr(self, "include_title_cb"):
            self.include_title_cb.setChecked(False)
            self.include_title_cb.toggled.connect(lambda _v: self._save_settings())
        if hasattr(self, "title_line"):
            self.title_line.textChanged.connect(lambda _v: self._save_settings())

    def _legend_anchor(self):
        txt = (
            getattr(self, "legend_corner_combo", None).currentText()
            if hasattr(self, "legend_corner_combo")
            else "Top Right"
        )
        mapping = {
            "Top Right": "top_right",
            "Top Left": "top_left",
            "Bottom Right": "bottom_right",
            "Bottom Left": "bottom_left",
        }
        return mapping.get(txt, "top_right")

    def _apply_labels_visibility(self):
        show = True
        if hasattr(self, "show_labels_cb"):
            show = self.show_labels_cb.isChecked()
        # Toggle labels on all interaction groups
        try:
            if show:
                cmd.show("labels", "Interactions")
            else:
                cmd.hide("labels", "Interactions")
        except Exception:
            pass

    def _refresh_ring_discs(self):
        # Recreate ring discs with new thickness for selected interactions
        try:
            for rid in list(self._selected_ids):
                try:
                    cmd.delete(f"intRingDiscProt_{rid}")
                    cmd.delete(f"intRingDiscLig_{rid}")
                except Exception:
                    pass
                # Recreate discs if interaction has ring info
                inter = self.interactions[rid]
                inter_type_group = f"Interactions.{inter['type'].replace(' ', '_')}"
                ring_color = COLOR_MAP.get(inter["type"], "yellow")
                if "prot_ring" in inter:
                    r = inter["prot_ring"]
                    self._make_ring_disc(
                        f"intRingDiscProt_{rid}", r["centroid"], r["normal"], r.get("radius", 1.8), ring_color
                    )
                    cmd.group(inter_type_group, f"intRingDiscProt_{rid}")
                if "ligand_ring" in inter:
                    r = inter["ligand_ring"]
                    self._make_ring_disc(
                        f"intRingDiscLig_{rid}", r["centroid"], r["normal"], r.get("radius", 1.8), ring_color
                    )
                    cmd.group(inter_type_group, f"intRingDiscLig_{rid}")
        except Exception:
            pass

    # --- Settings persistence ---
    def _settings_path(self):
        return os.path.join(os.path.dirname(__file__), "settings.json")

    def _load_settings(self):
        self._settings = {
            "legend_corner": "Top Right",
            "legend_export": True,
            "disc_thickness": 0.10,
            "show_labels": True,
            "max_distance": 0.0,
            "filter_selection": "All",
            "types_enabled": {},
            "compute_on_load": False,
            "auto_zoom": True,
            "confirm_remove_all": True,
            # chooser defaults
            "chooser_show": "All",
            "chooser_near": False,
            "chooser_near_dist": 6.0,
            "chooser_ref": "Protein",
            "chooser_ref_chain": "",
            "chooser_search": "",
        }
        try:
            p = self._settings_path()
            if os.path.exists(p):
                import json

                with open(p, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._settings.update(data)
        except Exception:
            pass

    def _save_settings(self):
        try:
            # Start from current settings to preserve non-UI flags
            data = dict(getattr(self, "_settings", {}))
            data.update(self._collect_settings_from_ui())
            import json

            with open(self._settings_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _apply_settings_to_ui(self):
        s = getattr(self, "_settings", {})
        try:
            if hasattr(self, "legend_corner_combo"):
                self.legend_corner_combo.blockSignals(True)
                self.legend_corner_combo.setCurrentText(s.get("legend_corner", "Top Right"))
                self.legend_corner_combo.blockSignals(False)
            if hasattr(self, "legend_export_cb"):
                self.legend_export_cb.blockSignals(True)
                self.legend_export_cb.setChecked(bool(s.get("legend_export", True)))
                self.legend_export_cb.blockSignals(False)
            if hasattr(self, "disc_thickness_spin"):
                self.disc_thickness_spin.blockSignals(True)
                self.disc_thickness_spin.setValue(float(s.get("disc_thickness", 0.10)))
                self.disc_thickness_spin.blockSignals(False)
            if hasattr(self, "show_labels_cb"):
                self.show_labels_cb.blockSignals(True)
                self.show_labels_cb.setChecked(bool(s.get("show_labels", True)))
                self.show_labels_cb.blockSignals(False)
            if hasattr(self, "max_distance_spin"):
                self.max_distance_spin.blockSignals(True)
                self.max_distance_spin.setValue(float(s.get("max_distance", 0.0)))
                self.max_distance_spin.blockSignals(False)
            if hasattr(self, "filter_combo"):
                self.filter_combo.blockSignals(True)
                self.filter_combo.setCurrentText(s.get("filter_selection", "All"))
                self.filter_combo.blockSignals(False)
            if hasattr(self, "legend_export_cb") and hasattr(self, "legend_corner_combo"):
                pass
            # Restore per-type toggles if present
            if hasattr(self, "_type_checkboxes"):
                for t, cb in self._type_checkboxes.items():
                    cb.blockSignals(True)
                    cb.setChecked(bool(s.get("types_enabled", {}).get(t, True)))
                    cb.blockSignals(False)
            # compute on load
            if hasattr(self, "compute_on_load_cb"):
                self.compute_on_load_cb.blockSignals(True)
                self.compute_on_load_cb.setChecked(bool(s.get("compute_on_load", False)))
                self.compute_on_load_cb.blockSignals(False)
            # auto zoom
            if hasattr(self, "auto_zoom_cb"):
                self.auto_zoom_cb.blockSignals(True)
                self.auto_zoom_cb.setChecked(bool(s.get("auto_zoom", True)))
                self.auto_zoom_cb.blockSignals(False)
            if hasattr(self, "show_angle_labels_cb"):
                self.show_angle_labels_cb.blockSignals(True)
                self.show_angle_labels_cb.setChecked(bool(s.get("show_angle_labels", False)))
                self.show_angle_labels_cb.blockSignals(False)
            if hasattr(self, "include_scale_cb"):
                self.include_scale_cb.blockSignals(True)
                self.include_scale_cb.setChecked(bool(s.get("include_scale", True)))
                self.include_scale_cb.blockSignals(False)
            if hasattr(self, "scale_length_spin"):
                self.scale_length_spin.blockSignals(True)
                self.scale_length_spin.setValue(int(s.get("scale_length", 10)))
                self.scale_length_spin.blockSignals(False)
            if hasattr(self, "include_title_cb"):
                self.include_title_cb.blockSignals(True)
                self.include_title_cb.setChecked(bool(s.get("include_title", False)))
                self.include_title_cb.blockSignals(False)
            if hasattr(self, "title_line"):
                self.title_line.blockSignals(True)
                self.title_line.setText(s.get("title_text", ""))
                self.title_line.blockSignals(False)
            # Apply side effects
            self._apply_labels_visibility()
            self._update_legend(for_export=False, anchor=self._legend_anchor())
            # Rebuild table with filters
            self._populate_table()
        except Exception:
            pass

    def _restore_last_session(self):
        s = getattr(self, "_settings", {})
        path = s.get("last_file")
        try:
            if path and os.path.isfile(path):
                # avoid file dialog; directly load
                self.pdb_path = path
                if hasattr(self, "file_edit"):
                    self.file_edit.setText(path)
                self._load_structure(path)
                base = os.path.splitext(os.path.basename(path))[0]
                cmd.reinitialize()
                self.loaded_object = base
                cmd.load(path, base)
                # Restore ligand fields if present
                lc = s.get("last_ligand_chain")
                lr = s.get("last_ligand_resi")
                ln = s.get("last_ligand_resn")
                if lc and lr and ln:
                    self.ligand_info = (lc, int(lr), ln)
                    self.ligand_chain.setText(lc)
                    self.ligand_resi.setText(str(lr))
                    self.ligand_resn.setText(ln)
                self._beautify_scene()
                # optionally auto-compute
                if bool(s.get("compute_on_load", False)) and self.ligand_info:
                    self._on_calc_interactions()
        except Exception:
            pass

    def _on_redraw_visible_types(self):
        try:
            for rid, inter in enumerate(self.interactions):
                itype = inter["type"]
                enabled = True
                if hasattr(self, "_type_checkboxes") and itype in self._type_checkboxes:
                    enabled = self._type_checkboxes[itype].isChecked()
                if enabled:
                    if rid in self._selected_ids:
                        # rebuild
                        self._toggle_interaction_by_id(rid)
                        self._toggle_interaction_by_id(rid)
                    else:
                        # create visuals
                        self._toggle_interaction_by_id(rid)
                else:
                    if rid in self._selected_ids:
                        # remove visuals
                        self._toggle_interaction_by_id(rid)
            self._apply_labels_visibility()
            self._update_legend(for_export=False, anchor=self._legend_anchor())
        except Exception:
            pass

    def _on_reset_defaults(self):
        try:
            if hasattr(self, "legend_corner_combo"):
                self.legend_corner_combo.setCurrentText("Top Right")
            if hasattr(self, "legend_export_cb"):
                self.legend_export_cb.setChecked(True)
            if hasattr(self, "disc_thickness_spin"):
                self.disc_thickness_spin.setValue(0.10)
            if hasattr(self, "show_labels_cb"):
                self.show_labels_cb.setChecked(True)
            self._save_settings()
            self._apply_labels_visibility()
            self._update_legend(for_export=False, anchor=self._legend_anchor())
            self._refresh_ring_discs()
        except Exception:
            pass

    def _collect_settings_from_ui(self):
        data = {}
        try:
            data["legend_corner"] = (
                self.legend_corner_combo.currentText() if hasattr(self, "legend_corner_combo") else "Top Right"
            )
            data["legend_export"] = (
                bool(self.legend_export_cb.isChecked()) if hasattr(self, "legend_export_cb") else True
            )
            data["disc_thickness"] = (
                float(self.disc_thickness_spin.value()) if hasattr(self, "disc_thickness_spin") else 0.10
            )
            data["show_labels"] = bool(self.show_labels_cb.isChecked()) if hasattr(self, "show_labels_cb") else True
            data["max_distance"] = float(self.max_distance_spin.value()) if hasattr(self, "max_distance_spin") else 0.0
            data["filter_selection"] = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All"
            # per-type toggles
            if hasattr(self, "_type_checkboxes"):
                data["types_enabled"] = {t: bool(cb.isChecked()) for t, cb in self._type_checkboxes.items()}
            # compute on load
            data["compute_on_load"] = (
                bool(self.compute_on_load_cb.isChecked()) if hasattr(self, "compute_on_load_cb") else False
            )
            data["auto_zoom"] = bool(self.auto_zoom_cb.isChecked()) if hasattr(self, "auto_zoom_cb") else True
            data["style_overrides"] = self._settings.get("style_overrides", {}) if hasattr(self, "_settings") else {}
            data["show_angle_labels"] = (
                bool(self.show_angle_labels_cb.isChecked()) if hasattr(self, "show_angle_labels_cb") else False
            )
            data["include_scale"] = (
                bool(self.include_scale_cb.isChecked()) if hasattr(self, "include_scale_cb") else True
            )
            data["scale_length"] = int(self.scale_length_spin.value()) if hasattr(self, "scale_length_spin") else 10
            data["include_title"] = (
                bool(self.include_title_cb.isChecked()) if hasattr(self, "include_title_cb") else False
            )
            data["title_text"] = self.title_line.text() if hasattr(self, "title_line") else ""
            data["confirm_remove_all"] = (
                bool(self._settings.get("confirm_remove_all", True)) if hasattr(self, "_settings") else True
            )
        except Exception:
            pass
        return data

    def _on_redraw_all(self):
        # Recreate visuals for all currently selected interactions
        try:
            for rid in list(self._selected_ids):
                # toggle off then on to fully rebuild visuals and apply per-type styles
                self._toggle_interaction_by_id(rid)
                self._toggle_interaction_by_id(rid)
            self._apply_labels_visibility()
            self._update_legend(for_export=False, anchor=self._legend_anchor())
        except Exception:
            pass

    def _on_apply_styles_now(self):
        # Apply color/style/opacity changes to all existing visuals without toggling selection state
        try:
            # Update legend to reflect colors first
            self._update_legend(for_export=False, anchor=self._legend_anchor())
            for rid, inter in enumerate(self.interactions):
                itype = inter["type"]
                color_name = self._type_color_name(itype)
                # Distance dashes
                dash = f"int_dash_{rid}"
                try:
                    cmd.set("dash_color", color_name, dash)
                    self._apply_dash_style(dash, itype)
                except Exception:
                    pass
                # Clouds
                for obj in (f"intCloudProt_{rid}", f"intCloudLig_{rid}"):
                    try:
                        cmd.color(color_name, obj)
                        ov = (
                            self._settings.get("style_overrides", {}).get(itype, {})
                            if hasattr(self, "_settings")
                            else {}
                        )
                        cov = ov.get("cloud_opacity")
                        trans = 1.0 - float(cov) if cov is not None else 0.6
                        cmd.set("transparency", trans, obj)
                    except Exception:
                        pass
                # Ring discs: rebuild with new thickness/opacity/colors if present
                for which, key, obj in (
                    ("prot", "prot_ring", f"intRingDiscProt_{rid}"),
                    ("lig", "ligand_ring", f"intRingDiscLig_{rid}"),
                ):
                    try:
                        cmd.delete(obj)
                    except Exception:
                        pass
                    try:
                        if key in inter:
                            r = inter[key]
                            self._make_ring_disc(obj, r["centroid"], r["normal"], r.get("radius", 1.8), itype)
                            group = f"Interactions.{itype.replace(' ', '_')}"
                            cmd.group(group, obj)
                    except Exception:
                        pass
                # Angle labels visibility toggling
                try:
                    lab = f"angle_label_{rid}"
                    if bool(self._settings.get("show_angle_labels", False)) if hasattr(self, "_settings") else False:
                        # Recreate at midpoint if dash exists
                        cmd.delete(lab)
                        detail = inter.get("details")
                        if detail and inter.get("prot_atom") and inter.get("lig_atom"):
                            prot_chain = inter["prot_res"].split()[1][0]
                            prot_resi = inter["prot_res"].split()[1][1:]
                            prot_res_sel = f"{self.loaded_object} and chain {prot_chain} and resi {prot_resi} and name {inter['prot_atom']}"
                            lig_chain, lig_resi, lig_resn = self.ligand_info
                            lig_atom_sel = f"{self.loaded_object} and chain {lig_chain} and resi {lig_resi} and name {inter['lig_atom']}"
                            pm = cmd.get_model(prot_res_sel)
                            lm = cmd.get_model(lig_atom_sel)
                            if pm.atom and lm.atom:
                                p = np.array(pm.atom[0].coord)
                                q = np.array(lm.atom[0].coord)
                                mid = (p + q) / 2.0
                                cmd.pseudoatom(lab, pos=[float(mid[0]), float(mid[1]), float(mid[2])])
                                cmd.label(lab, f'"{detail}"')
                                cmd.set("label_color", "white", lab)
                                cmd.set("label_outline_color", "black", lab)
                                cmd.group(f"Interactions.{itype.replace(' ', '_')}", lab)
                    else:
                        cmd.delete(lab)
                except Exception:
                    pass
        except Exception:
            pass

    def _populate_table_from(self, items):
        self.table.setRowCount(0)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Type", "Prot Res", "Prot Atom", "Lig Atom", "Distance (Å)", "Details"])
        self._table_items = list(items)
        for i, inter in enumerate(items):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(inter["type"]))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(inter["prot_res"]))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(inter["prot_atom"]))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(inter["lig_atom"]))
            self.table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{inter['distance']:.2f}"))
            self.table.setItem(i, 5, QtWidgets.QTableWidgetItem(inter.get("details", "")))
        self.table.resizeColumnsToContents()

    def _current_filtered_items(self):
        items = self.interactions
        # Type filter
        try:
            if hasattr(self, "filter_combo"):
                t = self.filter_combo.currentText()
                if t and t != "All":
                    items = [i for i in items if i["type"] == t]
        except Exception:
            pass
        # Distance filter
        try:
            if hasattr(self, "max_distance_spin"):
                md = float(self.max_distance_spin.value())
                if md > 0:
                    items = [i for i in items if i.get("distance", 0.0) <= md]
        except Exception:
            pass
        return items

    def _on_clear(self):
        # Remove all interaction visuals created by this session
        try:
            for rid in list(self._selected_ids):
                for prefix in (
                    "intRes_",
                    "int_dash_",
                    "intCloudProt_",
                    "intCloudLig_",
                    "intRingDiscProt_",
                    "intRingDiscLig_",
                ):
                    try:
                        cmd.delete(f"{prefix}{rid}")
                    except Exception:
                        pass
            self._selected_ids.clear()
        finally:
            pass

    def _on_remove_all_visuals(self):
        # Remove all Interactions.* groups and any int* objects, independent of selection
        try:
            if not self._confirm_remove_all():
                return
            # Delete group trees if present
            for grp in ("Interactions", "Interactions.Legend", "Interactions.ScaleBar", "Interactions.Title"):
                try:
                    cmd.delete(grp)
                except Exception:
                    pass
            # Wildcard cleanup for any leftover objects
            for pat in (
                "intRes_*",
                "int_dash_*",
                "intCloudProt_*",
                "intCloudLig_*",
                "intRingDiscProt_*",
                "intRingDiscLig_*",
                "legend_*",
                "angle_label_*",
                "scale_label",
                "title_label",
            ):
                try:
                    cmd.delete(pat)
                except Exception:
                    pass
            self._selected_ids.clear()
        except Exception:
            pass

    def _confirm_remove_all(self):
        # Show a confirmation dialog with "Don't ask again"
        try:
            if not bool(self._settings.get("confirm_remove_all", True)):
                return True
        except Exception:
            return True
        mbox = QtWidgets.QMessageBox(self)
        mbox.setIcon(QtWidgets.QMessageBox.Warning)
        mbox.setWindowTitle("Remove All Visuals")
        mbox.setText("Remove all plugin-generated visuals?")
        mbox.setInformativeText("This will delete all Interactions.* groups and int* objects. This cannot be undone.")
        mbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        mbox.setDefaultButton(QtWidgets.QMessageBox.No)
        chk = QtWidgets.QCheckBox("Don't ask again")
        mbox.setCheckBox(chk)
        ret = mbox.exec_()
        if ret == QtWidgets.QMessageBox.Yes:
            if chk.isChecked():
                try:
                    self._settings["confirm_remove_all"] = False
                    self._save_settings()
                except Exception:
                    pass
            return True
        return False

    def _on_export_csv(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export CSV", "interactions.csv", "CSV (*.csv)")
        if not out:
            return
        try:
            import csv

            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Type", "Prot Res", "Prot Atom", "Lig Atom", "Distance (Å)", "Details"])
                for i in self.interactions:
                    w.writerow(
                        [
                            i["type"],
                            i["prot_res"],
                            i["prot_atom"],
                            i["lig_atom"],
                            f"{i['distance']:.2f}",
                            i.get("details", ""),
                        ]
                    )
            QtWidgets.QMessageBox.information(self, "Exported", f"CSV saved to {out}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export CSV: {e}")

    def _update_legend(self, for_export=False, anchor="top_right"):
        # Explicit request to remove all legend objects: "legend_0" to "legend_9"
        try:
            cmd.delete("Interactions.Legend")
            # Loop arbitrary range to be safe
            for i in range(20):
                 cmd.delete(f"legend_{i}")
        except Exception:
            pass
            idx += 1
