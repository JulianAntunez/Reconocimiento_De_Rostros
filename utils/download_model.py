"""
Utilidad para descarga y verificación del modelo ONNX de reconocimiento de emociones (FER+).
"""

import sys
import urllib.request
from pathlib import Path

# Asegurar importación de módulos del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_CONFIG

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"


def descargar_modelo(destino: Path = DEFAULT_CONFIG.onnx_model_path) -> bool:
    """Descarga el modelo ONNX si no existe en la ruta de destino."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    if destino.exists() and destino.stat().st_size > 30_000_000:
        print(f"[OK] Modelo existente en: {destino} ({destino.stat().st_size / (1024 * 1024):.1f} MB)")
        return True

    print(f"Descargando modelo FER+ ONNX desde:\n  {MODEL_URL}")
    print(f"Destino: {destino} ...")

    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(destino, "wb") as f_out:
            total_bytes = int(resp.headers.get("Content-Length", 0))
            descargados = 0
            bloque = 1024 * 64

            while True:
                chunk = resp.read(bloque)
                if not chunk:
                    break
                f_out.write(chunk)
                descargados += len(chunk)
                if total_bytes > 0:
                    porcentaje = (descargados / total_bytes) * 100
                    print(f"\rProgreso: {porcentaje:.1f}% ({descargados / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB)", end="")

        print(f"\n[OK] Modelo descargado con éxito en: {destino}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Falló la descarga del modelo: {e}")
        if destino.exists():
            destino.unlink()
        return False


if __name__ == "__main__":
    exito = descargar_modelo()
    sys.exit(0 if exito else 1)
