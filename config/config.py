"""
Módulo de Configuración Centralizada
Define todos los parámetros del sistema (cámara, umbrales, rutas, periodicidad).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class AppConfig:
    """Configuración inmutable de la aplicación de reconocimiento de emociones."""

    # Dispositivo de captura
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # Umbrales de confianza (0.0 a 1.0)
    face_detection_confidence: float = 0.5
    emotion_confidence_threshold: float = 0.45  # Menor a este valor se cataloga como 'incierto'

    # Optimización de rendimiento (inferencia cada N frames)
    classify_every_n_frames: int = 3

    # Rutas de almacenamiento
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "logs")
    onnx_model_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "models" / "emotion_ferplus.onnx"
    )

    # Etiquetas de emociones en español (mapeo estándar FER-2013 / FER+)
    # Orden estándar: 0: neutral, 1: feliz, 2: sorpresa, 3: triste, 4: enojo, 5: asco, 6: miedo
    emotion_labels: List[str] = field(
        default_factory=lambda: [
            "neutral",
            "feliz",
            "sorprendido",
            "triste",
            "enojado",
            "asco",
            "miedo",
        ]
    )

    # Privacidad y seguridad de datos
    save_face_images: bool = False  # Por defecto NUNCA guardar imágenes de rostros


# Instancia por defecto para importar directamente en otros módulos
DEFAULT_CONFIG = AppConfig()
