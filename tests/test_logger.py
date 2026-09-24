"""
Pruebas unitarias para el registrador en CSV (Fase 4).
"""

from datetime import datetime, timezone
from pathlib import Path
import csv
import pytest
from core.classifier import EmotionResult
from utils import EmotionCSVLogger, CSV_COLUMNS


@pytest.fixture
def temp_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def sample_emotion_result():
    return EmotionResult(
        emotion="feliz",
        confidence=0.9250,
        raw_emotion="feliz",
        probabilities={
            "neutral": 0.02,
            "feliz": 0.925,
            "sorprendido": 0.03,
            "triste": 0.01,
            "enojado": 0.005,
            "asco": 0.005,
            "miedo": 0.005,
        },
        inference_time_ms=12.45,
    )


def test_logger_creacion_archivo_y_headers(temp_log_dir):
    """Verifica que el archivo CSV se cree con las columnas esperadas en el encabezado."""
    csv_file = "test_run.csv"
    with EmotionCSVLogger(output_dir=temp_log_dir, filename=csv_file, buffer_size=1) as logger:
        pass

    target_path = temp_log_dir / csv_file
    assert target_path.is_file()

    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == CSV_COLUMNS


def test_logger_registro_datos(temp_log_dir, sample_emotion_result):
    """Verifica que los datos del registro se guarden fielmente con formato ISO 8601 y 7 probabilidades."""
    csv_file = "test_records.csv"
    fixed_ts = datetime(2026, 9, 24, 20, 0, 0, tzinfo=timezone.utc)

    with EmotionCSVLogger(output_dir=temp_log_dir, filename=csv_file, source="webcam", buffer_size=1) as logger:
        ok = logger.log_prediction(
            frame_id=42,
            face_id=0,
            result=sample_emotion_result,
            custom_timestamp=fixed_ts,
        )
        assert ok is True
        assert logger.total_logged == 1

    target_path = temp_log_dir / csv_file
    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filas = list(reader)
        assert len(filas) == 1
        fila = filas[0]

        assert fila["timestamp"] == fixed_ts.isoformat()
        assert fila["source"] == "webcam"
        assert int(fila["frame_id"]) == 42
        assert int(fila["face_id"]) == 0
        assert fila["emotion"] == "feliz"
        assert float(fila["confidence"]) == 0.9250
        assert float(fila["prob_feliz"]) == 0.9250
        assert float(fila["prob_neutral"]) == 0.0200
        assert float(fila["latency_ms"]) == 12.45


def test_logger_buffer_y_flush(temp_log_dir, sample_emotion_result):
    """Verifica que el buffer almacene registros en memoria y solo escriba a disco al llenarse o al hacer flush."""
    csv_file = "test_buffer.csv"
    logger = EmotionCSVLogger(output_dir=temp_log_dir, filename=csv_file, buffer_size=5)

    # Añadir 3 registros (menos que el buffer_size=5)
    for i in range(3):
        logger.log_prediction(frame_id=i, face_id=0, result=sample_emotion_result)

    target_path = temp_log_dir / csv_file
    with open(target_path, mode="r", encoding="utf-8") as f:
        lineas_iniciales = f.readlines()
        # Solo debería estar el header
        assert len(lineas_iniciales) == 1

    # Hacer flush manual
    logger.flush()
    with open(target_path, mode="r", encoding="utf-8") as f:
        lineas_post_flush = f.readlines()
        assert len(lineas_post_flush) == 4  # 1 header + 3 registros

    logger.close()


def test_logger_manejo_resultado_invalido(temp_log_dir):
    """Verifica que pasar None no rompa la aplicación y devuelva False."""
    with EmotionCSVLogger(output_dir=temp_log_dir, filename="test_none.csv") as logger:
        res = logger.log_prediction(frame_id=1, face_id=0, result=None)
        assert res is False
        assert logger.total_logged == 0
