"""
Pruebas unitarias para el clasificador de emociones faciales (ONNX Runtime / FER+).
"""

import numpy as np
import pytest
from core import EmotionClassifier, EmotionResult
from config import DEFAULT_CONFIG


@pytest.fixture
def classifier():
    clf = EmotionClassifier(min_confidence=0.45)
    yield clf
    clf.close()


def test_classifier_inicializacion(classifier):
    """Verifica que el clasificador se inicialice y cargue el modelo ONNX."""
    assert classifier is not None
    assert classifier.min_confidence == 0.45
    assert len(classifier.labels) == 7
    assert classifier._session is not None


def test_classifier_input_invalido(classifier):
    """Verifica que entradas None, vacías o demasiado pequeñas no causen excepciones y devuelvan None."""
    assert classifier.predict(None) is None
    assert classifier.predict(np.array([])) is None
    assert classifier.predict(np.zeros((4, 4, 3), dtype=np.uint8)) is None


def test_classifier_prediccion_sintetica(classifier):
    """Verifica que un recorte sintético retorne una estructura EmotionResult válida."""
    fake_face = np.full((120, 120, 3), 128, dtype=np.uint8)
    res = classifier.predict(fake_face)

    assert isinstance(res, EmotionResult)
    assert res.emotion in classifier.labels or res.emotion == "incierto"
    assert res.raw_emotion in classifier.labels
    assert 0.0 <= res.confidence <= 1.0
    assert res.inference_time_ms >= 0.0
    assert len(res.probabilities) == 7


def test_classifier_probabilidades_suman_uno(classifier):
    """Verifica que las probabilidades de las 7 clases calculadas con Softmax sumen 1.0 (100%)."""
    fake_face = np.random.randint(50, 200, (96, 96, 3), dtype=np.uint8)
    res = classifier.predict(fake_face)

    assert res is not None
    suma_probs = sum(res.probabilities.values())
    assert pytest.approx(suma_probs, rel=1e-3) == 1.0


def test_classifier_umbral_incierto():
    """Verifica que si la confianza máxima no supera el umbral configurado, la emoción se marque como 'incierto'."""
    # Con un umbral imposible (0.999), cualquier predicción debe marcarse como 'incierto'
    clf_estricto = EmotionClassifier(min_confidence=0.999)
    fake_face = np.full((100, 100, 3), 120, dtype=np.uint8)
    res = clf_estricto.predict(fake_face)

    assert res is not None
    assert res.emotion == "incierto"
    # La emoción original sin filtrar debe seguir registrada en raw_emotion
    assert res.raw_emotion in clf_estricto.labels
    clf_estricto.close()


def test_classifier_escala_de_grises(classifier):
    """Verifica que el clasificador acepte imágenes en escala de grises directamente (2 dimensiones)."""
    fake_gray = np.full((80, 80), 100, dtype=np.uint8)
    res = classifier.predict(fake_gray)
    assert res is not None
    assert res.raw_emotion in classifier.labels
