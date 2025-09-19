# ProtLigInteract/__init__.py

from pymol.plugins import addmenuitemqt
from .code_v2 import ProtLigInteractDialog

__version__ = "1.0"
_dialog = None

def __init_plugin__(app=None):
    """
    PyMOL entry point: adds a menu item "Protein-Ligand Interactions"
    that launches our dialog.
    """
    global _dialog

    def run_plugin_gui():
        global _dialog
        if _dialog is not None:
            try:
                _dialog.close()
                _dialog.deleteLater()
            except Exception:
                pass
        _dialog = ProtLigInteractDialog()
        _dialog.show()

    addmenuitemqt("Protein-Ligand Interactions", run_plugin_gui)
