"""
Pruebas unitarias para la interfaz OpenCVApp (Fase 5).
"""

from pathlib import Path
import numpy as np
import pytest
from core.classifier import EmotionResult
from core.detector import FaceDetection
from interfaces import OpenCVApp


@pytest.fixture
def app():
    app_instance = OpenCVApp(enable_logging=False)
    yield app_instance
    app_instance.close()


def test_opencv_app_inicializacion(app):
    """Verifica que la app OpenCV se instancie con los modelos cargados."""
    assert app is not None
    assert app.detector is not None
    assert app.classifier is not None
    assert app.show_hud_bars is True
    assert app.show_stats is True


def test_opencv_app_notificaciones(app):
    """Verifica la asignación y expiración de notificaciones en pantalla."""
    app._set_notification("Prueba de Notificación", duration=1.0)
    assert app.notification_text == "Prueba de Notificación"
    assert app.notification_expiry > 0


def test_opencv_app_renderizar_frame(app):
    """Verifica que el renderizado de frame no arroje excepciones sobre una imagen sintética."""
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = FaceDetection(box=(50, 50, 100, 100), confidence=0.95)
    em_res = EmotionResult(
        emotion="feliz",
        confidence=0.90,
        raw_emotion="feliz",
        probabilities={"neutral": 0.05, "feliz": 0.90, "sorprendido": 0.05},
        inference_time_ms=10.0,
    )

    # Renderizar sin errores
    app._renderizar_frame(fake_frame, [(det, em_res)], fps=30.0, latencia_total=20.0)
    assert fake_frame.shape == (480, 640, 3)


def test_opencv_app_guardar_captura(app):
    """Verifica que la captura manual guarde el archivo JPG en disco y luego lo limpie."""
    fake_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    saved_path = app._guardar_captura_manual(fake_frame)

    p = Path(saved_path)
    assert p.is_file()
    assert p.stat().st_size > 0

    # Limpieza
    p.unlink()
