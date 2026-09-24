"""
Módulo de Registro y Telemetría CSV (Fase 4)
Guarda las predicciones de emociones y telemetría de rendimiento en formato tabular CSV
cumpliendo con la norma ISO 8601, manejo de sesiones y estrictas políticas de privacidad.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import csv
import uuid

from config import DEFAULT_CONFIG

if TYPE_CHECKING:
    from core.classifier import EmotionResult


# Columnas oficiales del archivo de registro CSV
CSV_COLUMNS = [
    "timestamp",
    "session_id",
    "source",
    "frame_id",
    "face_id",
    "emotion",
    "confidence",
    "prob_neutral",
    "prob_feliz",
    "prob_sorprendido",
    "prob_triste",
    "prob_enojado",
    "prob_asco",
    "prob_miedo",
    "latency_ms",
]


class EmotionCSVLogger:
    """
    Registrador en tiempo real de predicciones faciales a archivo CSV.
    Diseñado para alta frecuencia (30-60 FPS) con buffer eficiente y tolerancia a bloqueos de I/O.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        source: str = "webcam",
        session_id: Optional[str] = None,
        buffer_size: int = 10,
    ) -> None:
        """
        Inicializa el registrador CSV.

        :param output_dir: Directorio de almacenamiento de logs (por defecto data/logs).
        :param filename: Nombre opcional del archivo. Si es None, genera 'emociones_YYYYMMDD.csv'.
        :param source: Fuente de los frames ('webcam' o 'imagen').
        :param session_id: Identificador único de sesión. Si es None, genera un UUIDv4 corto.
        :param buffer_size: Cantidad de registros a mantener en memoria antes de hacer flush a disco.
        """
        self.output_dir = Path(output_dir or DEFAULT_CONFIG.logs_dir)
        self.source = str(source).lower()
        self.buffer_size = max(1, int(buffer_size))
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"

        self._records_buffer: List[List[Union[str, float, int]]] = []
        self._total_logged: int = 0
        self._file_handle = None
        self._csv_writer = None
        self._is_closed: bool = False

        # Generar nombre de archivo diario si no se proveyó uno específico
        ahora = datetime.now(timezone.utc)
        if not filename:
            filename = f"emociones_{ahora.strftime('%Y%m%d')}.csv"
        self.file_path = self.output_dir / filename

        self._iniciar_archivo()

    def _iniciar_archivo(self) -> None:
        """Crea el directorio y abre el archivo CSV en modo anexar (append), escribiendo headers si es nuevo."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            archivo_existe = self.file_path.exists() and self.file_path.stat().st_size > 0

            # Abrir con encoding utf-8 y newline='' según recomendación de la doc oficial de csv
            self._file_handle = open(self.file_path, mode="a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._file_handle)

            if not archivo_existe:
                self._csv_writer.writerow(CSV_COLUMNS)
                self._file_handle.flush()

        except (PermissionError, OSError) as e:
            print(f"[AVISO] No se pudo abrir el archivo CSV '{self.file_path}' para escritura: {e}")
            print("         Las predicciones se almacenarán temporalmente en memoria sin interrumpir el video.")
            self._file_handle = None
            self._csv_writer = None

    def log_prediction(
        self,
        frame_id: int,
        face_id: int,
        result: EmotionResult,
        custom_timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Registra una predicción de emoción de un rostro específico en un frame.

        :param frame_id: Número secuencial del frame en la sesión.
        :param face_id: Índice del rostro dentro del frame (0, 1, ...).
        :param result: Objeto EmotionResult devuelto por EmotionClassifier.
        :param custom_timestamp: Timestamp opcional para pruebas unitarias deterministas.
        :return: True si se encoló/registró con éxito, False en caso de error.
        """
        if self._is_closed:
            return False

        if result is None:
            return False

        ts = (custom_timestamp or datetime.now(timezone.utc)).isoformat()
        probs = result.probabilities or {}

        fila = [
            ts,
            self.session_id,
            self.source,
            int(frame_id),
            int(face_id),
            str(result.emotion),
            round(float(result.confidence), 4),
            round(float(probs.get("neutral", 0.0)), 4),
            round(float(probs.get("feliz", 0.0)), 4),
            round(float(probs.get("sorprendido", 0.0)), 4),
            round(float(probs.get("triste", 0.0)), 4),
            round(float(probs.get("enojado", 0.0)), 4),
            round(float(probs.get("asco", 0.0)), 4),
            round(float(probs.get("miedo", 0.0)), 4),
            round(float(result.inference_time_ms), 2),
        ]

        self._records_buffer.append(fila)
        self._total_logged += 1

        if len(self._records_buffer) >= self.buffer_size:
            self.flush()

        return True

    def flush(self) -> None:
        """Descarga el buffer en memoria hacia el archivo físico en disco."""
        if not self._records_buffer:
            return

        if self._file_handle is None or self._csv_writer is None:
            # Reintentar abrir por si el archivo estaba bloqueado transitoriamente (ej. por Excel)
            self._iniciar_archivo()
            if self._file_handle is None:
                return

        try:
            self._csv_writer.writerows(self._records_buffer)
            self._file_handle.flush()
            self._records_buffer.clear()
        except (PermissionError, OSError) as e:
            print(f"[AVISO] Falló el vaciado al CSV (posible archivo bloqueado): {e}")

    @property
    def total_logged(self) -> int:
        """Retorna la cantidad total de registros procesados en esta sesión."""
        return self._total_logged

    def close(self) -> None:
        """Vacía el buffer y cierra el descriptor de archivo de forma segura."""
        if self._is_closed:
            return

        self.flush()
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
            self._csv_writer = None

        self._is_closed = True

    def __enter__(self) -> "EmotionCSVLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
