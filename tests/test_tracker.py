"""
Pruebas unitarias para FaceTracker y TemporalSmoother (Fase 8).
"""

import pytest
from core.classifier import EmotionResult
from core.detector import FaceDetection
from core.tracker import FaceTracker, calcular_iou, distancia_centroides


def test_calcular_iou():
    """Verifica el cálculo de Intersección sobre Unión entre recuadros."""
    boxA = (10, 10, 50, 50)
    # Recuadros idénticos
    assert calcular_iou(boxA, boxA) == 1.0

    # Recuadros disjuntos
    boxB = (100, 100, 50, 50)
    assert calcular_iou(boxA, boxB) == 0.0

    # Recuadros con solapamiento parcial
    boxC = (35, 10, 50, 50)
    iou = calcular_iou(boxA, boxC)
    assert 0.0 < iou < 1.0


def test_tracker_asigna_id_persistente():
    """Verifica que un rostro en movimiento suave mantenga su track_id entre frames."""
    tracker = FaceTracker(window_size=5)

    det1 = FaceDetection(box=(50, 50, 60, 60), confidence=0.95)
    res1 = EmotionResult(
        emotion="feliz",
        confidence=0.85,
        raw_emotion="feliz",
        probabilities={"feliz": 0.85, "neutral": 0.15},
        inference_time_ms=10.0,
    )

    # Frame 1
    tracks_f1 = tracker.update([det1], [res1])
    assert len(tracks_f1) == 1
    tid1 = tracks_f1[0][0]

    # Frame 2: El rostro se movió 4 píxeles (continuidad espacial)
    det2 = FaceDetection(box=(54, 52, 60, 60), confidence=0.94)
    res2 = EmotionResult(
        emotion="feliz",
        confidence=0.88,
        raw_emotion="feliz",
        probabilities={"feliz": 0.88, "neutral": 0.12},
        inference_time_ms=10.0,
    )

    tracks_f2 = tracker.update([det2], [res2])
    assert len(tracks_f2) == 1
    tid2 = tracks_f2[0][0]

    # El ID debe ser idéntico
    assert tid1 == tid2


def test_tracker_nuevo_rostro():
    """Verifica que un rostro en una posición completamente nueva reciba un track_id nuevo."""
    tracker = FaceTracker()

    det1 = FaceDetection(box=(20, 20, 40, 40), confidence=0.9)
    res1 = EmotionResult(emotion="neutral", confidence=0.8, raw_emotion="neutral", probabilities={"neutral": 0.8}, inference_time_ms=10.0)
    tracks_f1 = tracker.update([det1], [res1])
    id_cara_1 = tracks_f1[0][0]

    # Aparece un segundo rostro en el otro extremo de la pantalla
    det2 = FaceDetection(box=(400, 300, 50, 50), confidence=0.92)
    res2 = EmotionResult(emotion="feliz", confidence=0.9, raw_emotion="feliz", probabilities={"feliz": 0.9}, inference_time_ms=10.0)

    tracks_f2 = tracker.update([det1, det2], [res1, res2])
    assert len(tracks_f2) == 2
    ids_f2 = [t[0] for t in tracks_f2]

    assert id_cara_1 in ids_f2
    assert len(set(ids_f2)) == 2


def test_suavizado_temporal_elimina_flicker():
    """Verifica que una perturbación de 1 solo frame no cambie la emoción estabilizada."""
    tracker = FaceTracker(window_size=5)
    box = (100, 100, 80, 80)

    # 4 frames consecutivos claramente "neutral"
    res_neutral = EmotionResult(
        emotion="neutral",
        confidence=0.90,
        raw_emotion="neutral",
        probabilities={"neutral": 0.90, "enojado": 0.10},
        inference_time_ms=10.0,
    )
    for _ in range(4):
        tracker.update([FaceDetection(box=box, confidence=0.9)], [res_neutral])

    # 1 frame anómalo (flicker aislado) que dice "enojado" con 60%
    res_glitch = EmotionResult(
        emotion="enojado",
        confidence=0.60,
        raw_emotion="enojado",
        probabilities={"neutral": 0.40, "enojado": 0.60},
        inference_time_ms=10.0,
    )
    res_final = tracker.update([FaceDetection(box=box, confidence=0.9)], [res_glitch])[0][2]

    # Debido al suavizado de los 5 frames, "neutral" sigue siendo la emoción dominante
    assert res_final.emotion == "neutral"
    assert res_final.probabilities["neutral"] > res_final.probabilities["enojado"]


def test_tracker_limpieza_tracks_perdidos():
    """Verifica que tras max_lost frames sin detección, el track inactivo sea purgado."""
    tracker = FaceTracker(max_lost=3)
    det = FaceDetection(box=(100, 100, 50, 50), confidence=0.9)
    res = EmotionResult(emotion="neutral", confidence=0.8, raw_emotion="neutral", probabilities={"neutral": 0.8}, inference_time_ms=10.0)

    tracker.update([det], [res])
    assert len(tracker.tracks) == 1

    # 3 frames sin rostros
    for _ in range(3):
        tracker.update([], [])
    assert len(tracker.tracks) == 1  # Todavía en tolerancia

    # 4to frame sin rostros: excede max_lost=3
    tracker.update([], [])
    assert len(tracker.tracks) == 0  # Eliminado de memoria
