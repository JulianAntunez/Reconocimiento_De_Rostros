"""
Pruebas unitarias para el detector de rostros (MediaPipe BlazeFace).
"""

import numpy as np
import pytest
from core import FaceDetector, FaceDetection


@pytest.fixture
def detector():
    det = FaceDetector(min_confidence=0.5)
    yield det
    det.close()


def test_detector_inicializacion(detector):
    """Verifica que el detector se inicialice correctamente."""
    assert detector is not None
    assert detector.min_confidence == 0.5


def test_detector_frame_vacio(detector):
    """Verifica que un frame None o vacío devuelva lista vacía sin arrojar excepciones."""
    assert detector.detect(None) == []
    assert detector.detect(np.array([])) == []


def test_detector_imagen_negra(detector):
    """Verifica que una imagen sintética sin caras retorne lista vacía de detecciones."""
    imagen_negra = np.zeros((480, 640, 3), dtype=np.uint8)
    detecciones = detector.detect(imagen_negra)
    assert isinstance(detecciones, list)
    assert len(detecciones) == 0


def test_crop_face_limites():
    """Verifica que la función crop_face respete los límites de la imagen y no desborde."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Recorte normal dentro de los límites
    crop = FaceDetector.crop_face(frame, box=(20, 20, 40, 40), margin=0.1)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0

    # Recorte en los bordes extremos (debe recortar sin error de índice)
    crop_borde = FaceDetector.crop_face(frame, box=(0, 0, 50, 50), margin=0.5)
    assert crop_borde is not None
    assert crop_borde.shape[0] <= 100 and crop_borde.shape[1] <= 100

    # Frame inválido
    assert FaceDetector.crop_face(None, (0, 0, 10, 10)) is None
