"""
Punto de entrada para despliegues web (Streamlit Cloud, Hugging Face Spaces).
Ejecuta la interfaz web interactiva ubicada en interfaces/streamlit_app.py
"""

from pathlib import Path
import runpy
import sys

# Asegurar raíz del proyecto en sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ejecutar el dashboard de Streamlit
target_app = ROOT_DIR / "interfaces" / "streamlit_app.py"
runpy.run_path(str(target_app), run_name="__main__")
