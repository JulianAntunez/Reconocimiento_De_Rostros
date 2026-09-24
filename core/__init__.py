"""
Módulo del núcleo (core) del sistema de reconocimiento de rostros y emociones.
"""

from .detector import FaceDetector, FaceDetection
from .classifier import EmotionClassifier, EmotionResult
from .tracker import FaceTracker, TrackedFace, calcular_iou

__all__ = [
    "FaceDetector",
    "FaceDetection",
    "EmotionClassifier",
    "EmotionResult",
    "FaceTracker",
    "TrackedFace",
    "calcular_iou",
]
