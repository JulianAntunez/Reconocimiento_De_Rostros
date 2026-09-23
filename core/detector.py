"""
Módulo de Detección de Rostros
Utiliza MediaPipe Face Detection (BlazeFace) para detección ultra-rápida y precisa en CPU.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceDetection:
    """Estructura de datos con la información de un rostro detectado."""

    box: Tuple[int, int, int, int]  # (x, y, ancho, alto) en píxeles absolutos
    confidence: float  # Confianza de la detección entre 0.0 y 1.0
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None  # Puntos clave (ojos, nariz, boca)


class FaceDetector:
    """Detector de rostros en tiempo real basado en MediaPipe BlazeFace."""

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 0) -> None:
        """
        Inicializa el detector de rostros.

        :param min_confidence: Umbral mínimo de confianza para considerar una detección válida.
        :param model_selection: 0 para rostros a corta distancia (<= 2 metros, ideal webcam),
                                1 para rostros de rango completo (<= 5 metros).
        """
        self.min_confidence = float(min_confidence)
        self.model_selection = int(model_selection)

        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_confidence,
            model_selection=self.model_selection,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detecta todos los rostros presentes en un frame de video o imagen.

        :param frame: Imagen en formato BGR de OpenCV (NumPy array).
        :return: Lista de objetos FaceDetection encontrados. Si no hay rostros, retorna lista vacía [].
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return []

        # MediaPipe requiere formato RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = self._detector.process(rgb_frame)
        except Exception as e:
            print(f"[ERROR] Error al procesar el frame en MediaPipe: {e}")
            return []

        if not results or not results.detections:
            return []

        detections: List[FaceDetection] = []

        for raw_det in results.detections:
            score = float(raw_det.score[0]) if raw_det.score else 0.0
            if score < self.min_confidence:
                continue

            rel_box = raw_det.location_data.relative_bounding_box
            # Conversión de coordenadas relativas a píxeles absolutos
            x = int(rel_box.xmin * w)
            y = int(rel_box.ymin * h)
            bw = int(rel_box.width * w)
            bh = int(rel_box.height * h)

            # Delimitar (clamping) dentro de las dimensiones de la imagen
            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(w, x + bw)
            y_max = min(h, y + bh)

            ancho_final = max(0, x_max - x_min)
            alto_final = max(0, y_max - y_min)

            if ancho_final < 10 or alto_final < 10:
                # Descartar recuadros degenerados o insignificantes (ruido)
                continue

            # Extracción opcional de puntos clave de landmarks
            landmarks_dict: Dict[str, Tuple[int, int]] = {}
            if raw_det.location_data.keypoints:
                nombres_puntos = [
                    "ojo_derecho",
                    "ojo_izquierdo",
                    "punta_nariz",
                    "centro_boca",
                    "trago_oreja_derecha",
                    "trago_oreja_izquierda",
                ]
                for i, kp in enumerate(raw_det.location_data.keypoints):
                    if i < len(nombres_puntos):
                        kp_x = int(kp.x * w)
                        kp_y = int(kp.y * h)
                        landmarks_dict[nombres_puntos[i]] = (kp_x, kp_y)

            detections.append(
                FaceDetection(
                    box=(x_min, y_min, ancho_final, alto_final),
                    confidence=score,
                    landmarks=landmarks_dict,
                )
            )

        return detections

    @staticmethod
    def crop_face(
        frame: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.15
    ) -> Optional[np.ndarray]:
        """
        Recorta la región de interés (ROI) del rostro aplicando un margen proporcional.

        :param frame: Imagen fuente en formato BGR.
        :param box: Coordenadas (x, y, w, h) en píxeles.
        :param margin: Margen porcentual a expandir alrededor del rostro (ej. 0.15 = 15%).
        :return: Sub-imagen recortada o None si las coordenadas son inválidas.
        """
        if frame is None or frame.size == 0:
            return None

        h_img, w_img = frame.shape[:2]
        x, y, w, h = box

        # Calcular expansión por margen
        dx = int(w * margin)
        dy = int(h * margin)

        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(w_img, x + w + dx)
        y2 = min(h_img, y + h + dy)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def close(self) -> None:
        """Libera los recursos de MediaPipe."""
        if hasattr(self, "_detector") and self._detector is not None:
            self._detector.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
