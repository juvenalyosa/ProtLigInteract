# ProtLigInteract/code_v2.py
import os
from collections import defaultdict

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from pymol import cgo, cmd
from pymol.Qt import QtCore, QtWidgets
from pymol.Qt.utils import loadUi

# --- Expanded color palette for new interaction types ---
COLOR_MAP = {
    "Hydrogen Bond": "cyan",
    "Salt Bridge": "red",
    "Hydrophobic": "green",
    "Pi-Pi Stacking": "orange",
    "T-Shaped Pi-Pi": "darkorange",
    "Cation-Pi": "magenta",
    "Halogen Bond": "tv_yellow",
    "Metal Coordination": "purple",
    "Van der Waals": "lightorange",
    "Selected": "yellow",
}

# Per-type dash styling (applied per distance object)
STYLE_MAP = {
    "Hydrogen Bond": {"dash_radius": 0.06, "dash_length": 0.35, "dash_gap": 0.15},
    "Halogen Bond": {"dash_radius": 0.07, "dash_length": 0.40, "dash_gap": 0.18},
    "Salt Bridge": {"dash_radius": 0.09, "dash_length": 0.55, "dash_gap": 0.22},
    "Metal Coordination": {"dash_radius": 0.07, "dash_length": 0.45, "dash_gap": 0.18},
    "Cation-Pi": {"dash_radius": 0.08, "dash_length": 0.45, "dash_gap": 0.20},
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
    "protein_positive": {"LYS": ["NZ"], "ARG": ["NH1", "NH2", "NE"], "HIS": ["ND1", "NE2"]},
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
            self.hist_btn.clicked.connect(self._on_show_hist)
        if hasattr(self, "edit_styles_btn"):
            self.edit_styles_btn.clicked.connect(self._on_edit_styles)
        if hasattr(self, "apply_styles_btn"):
            self.apply_styles_btn.clicked.connect(self._on_apply_styles_now)
        if hasattr(self, "remove_all_btn"):
            self.remove_all_btn.clicked.connect(self._on_remove_all_visuals)

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
        self.interactions = self._calculate_all_interactions()
        self._populate_table()
        self._beautify_scene()
        QtWidgets.QMessageBox.information(self, "Done", f"Found {len(self.interactions)} interactions.")
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

    def _precompute_atom_features(self):
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
        if features["ligand_atoms"]:
            features["ligand_centroid"] = get_centroid([a["coord"] for a in features["ligand_atoms"]])
        else:
            features["ligand_centroid"] = None

        return features

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

        features = self._precompute_atom_features()
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
        for p_d in features["protein_h_donors"]:
            for l_a in grid_lig_acc.query(p_d["coord"], hb_cut):
                if l_a["element"] in ("O", "N"):
                    ok, d, detail = is_hbond(p_d, l_a)
                    if ok:
                        add_interaction("Hydrogen Bond", p_d, l_a, d, detail)
        # Ligand donor -> Protein acceptor
        for l_d in features["ligand_h_donors_acceptors"]:
            for p_a in grid_prot_acc.query(l_d["coord"], hb_cut):
                if l_d["element"] in ("O", "N"):
                    ok, d, detail = is_hbond(l_d, p_a)
                    if ok:
                        add_interaction("Hydrogen Bond", p_a, l_d, d, detail)

        # --- Salt bridges ---
        sb_cut = GEOMETRY_CRITERIA["salt_bridge_dist"]
        grid_lig_oxyn = self._NeighborGrid(
            [a for a in features["ligand_atoms"] if a["element"] in ("O", "N")], cell_size=sb_cut
        )
        for p_pos in features["protein_positive"]:
            for l_neg in grid_lig_oxyn.query(p_pos["coord"], sb_cut):
                d = distance(p_pos["coord"], l_neg["coord"])
                if d <= sb_cut:
                    add_interaction("Salt Bridge", p_pos, l_neg, d)
        grid_lig_n = self._NeighborGrid([a for a in features["ligand_atoms"] if a["element"] == "N"], cell_size=sb_cut)
        for p_neg in features["protein_negative"]:
            for l_pos in grid_lig_n.query(p_neg["coord"], sb_cut):
                d = distance(p_neg["coord"], l_pos["coord"])
                if d <= sb_cut:
                    add_interaction("Salt Bridge", p_neg, l_pos, d)

        # --- Metal coordination ---
        mc_cut = GEOMETRY_CRITERIA["metal_coordination_dist"]
        grid_lig_macc = self._NeighborGrid(features["ligand_metal_acceptors"], cell_size=mc_cut)
        for metal in features["metals"]:
            for l_acc in grid_lig_macc.query(metal["coord"], mc_cut):
                d = distance(metal["coord"], l_acc["coord"])
                if d <= mc_cut:
                    add_interaction("Metal Coordination", metal, l_acc, d)

        # --- Halogen bonds with C–X...A angle ---
        hb_hal_cut = GEOMETRY_CRITERIA["halogen_dist"]
        grid_prot_acc2 = self._NeighborGrid(features["protein_h_acceptors"], cell_size=hb_hal_cut)
        for l_hal in features["ligand_halogens"]:
            # find bound carbon to halogen
            neighbors = self._neighbors(l_hal, features["ligand_atoms"], element_filter={"C"})
            carbon = neighbors[0] if neighbors else None
            for p_acc in grid_prot_acc2.query(l_hal["coord"], hb_hal_cut):
                d = distance(l_hal["coord"], p_acc["coord"])
                if d <= GEOMETRY_CRITERIA["halogen_dist"]:
                    detail = None
                    if carbon is not None:
                        ang = angle_between_vectors(carbon["coord"] - l_hal["coord"], p_acc["coord"] - l_hal["coord"])
                        if ang < GEOMETRY_CRITERIA["halogen_angle"]:
                            continue
                        detail = f"CXA: {ang:.1f}°"
                    add_interaction("Halogen Bond", p_acc, l_hal, d, detail)

        # --- Pi-System interactions ---
        # Cation-pi: cationic protein atom to ligand ring (if any) and vice versa
        cp_cut = GEOMETRY_CRITERIA["cation_pi_dist"]
        grid_lig_posN = self._NeighborGrid(
            [a for a in features["ligand_atoms"] if a["element"] == "N"], cell_size=cp_cut
        )
        for p_ring in features["protein_rings"]:
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
            # ring-ring
            grid_p_rings = self._NeighborGrid(features["protein_rings"], cell_size=GEOMETRY_CRITERIA["pi_t_dist"])
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
                        "Pi-Pi Stacking",
                        p_placeholder,
                        l_placeholder,
                        d,
                        f"Angle: {angle:.1f}°",
                        extra={"prot_ring": p_ring, "ligand_ring": l_ring},
                    )
                elif (
                    GEOMETRY_CRITERIA["pi_pi_dist"] < d <= GEOMETRY_CRITERIA["pi_t_dist"]
                    and GEOMETRY_CRITERIA["pi_t_angle_low"] <= angle <= GEOMETRY_CRITERIA["pi_t_angle_high"]
                ):
                    add_interaction(
                        "T-Shaped Pi-Pi",
                        p_placeholder,
                        l_placeholder,
                        d,
                        f"Angle: {angle:.1f}°",
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
                cmd.color("white", res_name)
                cmd.util.cba(20, res_name)  # Color by element for protein residue
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
            elif "Ring" not in prot_atom_names and "Ring" not in lig_atom_names:
                # --- Dash Visualization for Specific Interactions ---
                prot_atom_sel = f"({prot_res_sel} and name {prot_atom_names})"
                lig_atom_sel = (
                    f"({self.loaded_object} and chain {lig_chain} and resi {lig_resi} and name {lig_atom_names})"
                )
                try:
                    cmd.distance(dash_name, prot_atom_sel, lig_atom_sel)
                    cmd.set("dash_color", self._type_color_name(inter["type"]), dash_name)
                    self._apply_dash_style(dash_name, inter["type"])
                    cmd.group(inter_type_group, dash_name)
                    # Optional angle label
                    if bool(self._settings.get("show_angle_labels", False)) if hasattr(self, "_settings") else False:
                        detail = inter.get("details")
                        if detail:
                            try:
                                pm = cmd.get_model(prot_atom_sel)
                                lm = cmd.get_model(lig_atom_sel)
                                if pm.atom and lm.atom:
                                    p = np.array(pm.atom[0].coord)
                                    q = np.array(lm.atom[0].coord)
                                    mid = (p + q) / 2.0
                                    labname = f"angle_label_{rid}"
                                    cmd.pseudoatom(labname, pos=[float(mid[0]), float(mid[1]), float(mid[2])])
                                    cmd.label(labname, f'"{detail}"')
                                    cmd.set("label_color", "white", labname)
                                    cmd.set("label_outline_color", "black", labname)
                                    cmd.group(inter_type_group, labname)
                            except Exception:
                                pass
                except Exception:
                    pass

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

        # Set a black background
        cmd.bg_color("black")

        # Show polymer (protein/nucleic acids) as clean white cartoon, excluding selected ligand residue
        prot_sel = f"{self.loaded_object} and polymer"
        if self.ligand_info:
            c, i, r = self.ligand_info
            lig_excl = f" and not (chain {c} and resi {i} and resn {r})"
            prot_sel += lig_excl
        cmd.show("cartoon", prot_sel)
        cmd.color("white", prot_sel)

        # Show ligand in ball-and-stick, colored by element
        if self.ligand_info:
            c, i, r = self.ligand_info
            lig_sel = f"{self.loaded_object} and chain {c} and resi {i} and resn {r}"
            try:
                cmd.show("sticks", lig_sel)
                cmd.show("spheres", lig_sel)
                cmd.set("stick_radius", 0.15, lig_sel)
                cmd.set("sphere_scale", 0.25, lig_sel)
                cmd.util.cba(20, lig_sel)  # Color by element (CPK colors)
                if bool(self._settings.get("auto_zoom", True)) if hasattr(self, "_settings") else True:
                    cmd.zoom(lig_sel, 8)
            except Exception:
                pass

        # General settings for a clean look
        cmd.set("antialias", 2)
        cmd.set("dash_gap", 0.2)
        cmd.set("dash_length", 0.4)
        cmd.set("dash_radius", 0.08)
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
        # If UI has a legend_layout, populate it; safe to no-op otherwise
        legend_container = getattr(self, "legend_grid", None) or self.findChild(QtWidgets.QGridLayout, "legend_grid")
        if not legend_container:
            return
        # Clear existing
        while legend_container.count():
            item = legend_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        # Populate
        row = 0
        for name, color in COLOR_MAP.items():
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background:{color}; border:1px solid #333;")
            label = QtWidgets.QLabel(name)
            legend_container.addWidget(swatch, row, 0)
            legend_container.addWidget(label, row, 1)
            row += 1

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
        # Remove previous legend
        try:
            cmd.delete("Interactions.Legend")
        except Exception:
            pass
        # Position based on object extent for export-safe placement
        view = cmd.get_view()
        # object center and scale
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
        # offset as fraction of object size; increase for export
        frac = 0.35 if for_export else 0.25
        off_r = frac * diag
        off_u = frac * diag
        if anchor == "top_left":
            off_r = -off_r
        if anchor in ("bottom_left", "bottom_right"):
            off_u = -off_u
        base = np.array(center) + right * off_r + up * off_u
        dy = -0.12 * diag  # line spacing scales with object size

        idx = 0
        for name, color in COLOR_MAP.items():
            pos = base + up * (dy * idx)
            ps_name = f"legend_{idx}"
            try:
                cmd.pseudoatom(ps_name, pos=[float(pos[0]), float(pos[1]), float(pos[2])])
                cmd.show("spheres", ps_name)
                cmd.set("sphere_scale", 0.3, ps_name)
                cmd.color(self._type_color_name(name), ps_name)
                cmd.label(ps_name, f'"{name}"')
                cmd.set("label_color", "white", ps_name)
                cmd.set("label_outline_color", "black", ps_name)
                # offset label to the right
                offset = (right * (0.15 * diag)).tolist()
                cmd.set("label_position", [float(offset[0]), float(offset[1]), float(offset[2])], ps_name)
                cmd.group("Interactions.Legend", ps_name)
            except Exception:
                pass
            idx += 1
