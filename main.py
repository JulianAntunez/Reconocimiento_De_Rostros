"""
Punto de Entrada Principal del Sistema de Reconocimiento Facial y Clasificación de Emociones
"""

import argparse
import sys
from pathlib import Path

# Asegurar importación de módulos locales
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG
from interfaces import OpenCVApp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sistema de Detección de Rostros y Reconocimiento de Emociones en Tiempo Real (MediaPipe + ONNX FER+)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a una imagen estática (JPG o PNG) para procesar.",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=DEFAULT_CONFIG.camera_index,
        help="Índice de la cámara a utilizar (por defecto 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIG.emotion_confidence_threshold,
        help="Umbral de confianza para clasificar emociones (menor a este valor se cataloga como 'incierto').",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Desactiva el registro continuo en el archivo CSV de data/logs/.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Ejecuta el script de diagnóstico de entorno y hardware antes de iniciar.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Lanza el dashboard web interactivo con Streamlit en el navegador.",
    )

    args = parser.parse_args()

    if args.web:
        import subprocess
        streamlit_app = str(Path(__file__).resolve().parent / "interfaces" / "streamlit_app.py")
        print("\nIniciando Dashboard Web de Streamlit...")
        print("Se abrirá automáticamente en tu navegador predeterminado (http://localhost:8501)")
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_app])
        return

    if args.check:
        from check_environment import main as run_check
        print("Ejecutando diagnóstico inicial...")
        run_check()

    app = OpenCVApp(
        camera_index=args.cam,
        threshold=args.threshold,
        enable_logging=not args.no_log,
    )

    try:
        if args.image:
            app.run_image(args.image)
        else:
            app.run_webcam()
    finally:
        app.close()


if __name__ == "__main__":
    main()
