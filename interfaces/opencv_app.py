"""
Módulo de la Interfaz OpenCV Integrada (Fase 5)
Aplicación gráfica de escritorio en tiempo real que combina detección facial,
clasificación de emociones, telemetría de rendimiento y registro en CSV.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from config import DEFAULT_CONFIG, AppConfig
from core import FaceDetector, EmotionClassifier, EmotionResult, FaceDetection
from utils import EmotionCSVLogger

# Paleta armónica de colores BGR según la emoción
EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "feliz": (50, 205, 50),        # Verde lima
    "sorprendido": (0, 215, 255),  # Amarillo oro
    "neutral": (220, 220, 220),    # Blanco / Gris claro
    "triste": (255, 140, 0),       # Azul acero (en BGR: 0, 140, 255)
    "enojado": (30, 30, 255),      # Rojo vibrante
    "miedo": (180, 105, 255),      # Violeta / Magenta
    "asco": (0, 140, 0),           # Verde bosque
    "incierto": (130, 130, 130),   # Gris neutro
}


class OpenCVApp:
    """Aplicación principal de escritorio con interfaz gráfica en OpenCV."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        camera_index: Optional[int] = None,
        threshold: Optional[float] = None,
        enable_logging: bool = True,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.camera_index = camera_index if camera_index is not None else self.config.camera_index
        self.threshold = threshold if threshold is not None else self.config.emotion_confidence_threshold
        self.enable_logging = enable_logging

        # Inicialización de modelos
        self.detector = FaceDetector(min_confidence=self.config.face_detection_confidence)
        self.classifier = EmotionClassifier(min_confidence=self.threshold)

        # Estado de la interfaz
        self.show_hud_bars: bool = True
        self.show_stats: bool = True
        self.notification_text: str = ""
        self.notification_expiry: float = 0.0

        # Lista de umbrales cíclicos con tecla 't'
        self._threshold_levels = [0.35, 0.45, 0.60]
        self._current_thresh_idx = 1  # 0.45 por defecto

        # Directorio para capturas manuales bajo demanda
        self.captures_dir = self.config.project_root / "data" / "capturas"
        self.captures_dir.mkdir(parents=True, exist_ok=True)

        # Métricas de sesión
        self.session_counts: Dict[str, int] = {label: 0 for label in self.config.emotion_labels}
        self.session_counts["incierto"] = 0

    def _set_notification(self, text: str, duration: float = 2.5) -> None:
        """Configura un mensaje emergente temporal en pantalla."""
        self.notification_text = text
        self.notification_expiry = time.time() + duration

    def _dibujar_esquinas_box(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        longitud: int = 15,
    ) -> None:
        """Dibuja un recuadro con esquinas resaltadas estilo HUD."""
        x, y, w, h = box
        # Recuadro base más fino
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        l = min(longitud, w // 4, h // 4)
        t = thickness

        # Esquina superior izquierda
        cv2.line(frame, (x, y), (x + l, y), color, t)
        cv2.line(frame, (x, y), (x, y + l), color, t)

        # Esquina superior derecha
        cv2.line(frame, (x + w, y), (x + w - l, y), color, t)
        cv2.line(frame, (x + w, y), (x + w, y + l), color, t)

        # Esquina inferior izquierda
        cv2.line(frame, (x, y + h), (x + l, y + h), color, t)
        cv2.line(frame, (x, y + h), (x, y + h - l), color, t)

        # Esquina inferior derecha
        cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, t)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, t)

    def _dibujar_panel_probabilidades(
        self,
        frame: np.ndarray,
        result: EmotionResult,
        x_pos: int = 15,
        y_pos: int = 100,
        ancho_barra: int = 130,
        alto_barra: int = 13,
    ) -> None:
        """Dibuja el panel HUD con barras horizontales de probabilidades de las 7 clases."""
        num_emociones = len(result.probabilities)
        alto_panel = num_emociones * 20 + 35
        ancho_panel = ancho_barra + 130

        # Fondo translúcido
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_pos - 8, y_pos - 22),
            (x_pos + ancho_panel, y_pos + alto_panel - 15),
            (25, 25, 25),
            -1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(
            frame,
            "PROBABILIDADES (HUD):",
            (x_pos, y_pos - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )

        y_actual = y_pos + 12
        for label, prob in result.probabilities.items():
            color = EMOTION_COLORS.get(label, (255, 255, 255))
            es_top = (label == result.raw_emotion)

            # Nombre de la emoción
            cv2.putText(
                frame,
                f"{label[:8].capitalize()}:",
                (x_pos, y_actual + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color if es_top else (175, 175, 175),
                1,
            )

            # Barra de fondo
            bar_x = x_pos + 80
            cv2.rectangle(
                frame,
                (bar_x, y_actual),
                (bar_x + ancho_barra, y_actual + alto_barra),
                (55, 55, 55),
                -1,
            )

            # Barra coloreada proporcional
            ancho_lleno = int(ancho_barra * max(0.0, min(1.0, prob)))
            if ancho_lleno > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, y_actual),
                    (bar_x + ancho_lleno, y_actual + alto_barra),
                    color,
                    -1,
                )

            # Porcentaje
            cv2.putText(
                frame,
                f"{prob * 100:.1f}%",
                (bar_x + ancho_barra + 6, y_actual + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (255, 255, 255),
                1,
            )

            y_actual += 20

    def _renderizar_frame(
        self,
        frame: np.ndarray,
        detecciones_con_emocion: List[Tuple[FaceDetection, EmotionResult]],
        fps: float,
        latencia_total: float,
    ) -> None:
        """Dibuja todos los elementos gráficos sobre el frame."""
        # 1. Recuadros y etiquetas de rostros
        for idx, (det, em_res) in enumerate(detecciones_con_emocion, start=1):
            x, y, w, h = det.box
            color = EMOTION_COLORS.get(em_res.emotion, (0, 255, 0))

            # Dibujar esquinas del recuadro
            self._dibujar_esquinas_box(frame, det.box, color=color, thickness=2)

            # Etiqueta de emoción con fondo
            etiqueta = f"#{idx} {em_res.emotion.upper()}: {em_res.confidence * 100:.0f}%"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
            y_text = max(26, y - 8)

            cv2.rectangle(
                frame,
                (x, y_text - th - 6),
                (x + tw + 8, y_text + 4),
                color,
                -1,
            )
            cv2.putText(
                frame,
                etiqueta,
                (x + 4, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 0),
                2,
            )

        # 2. Panel HUD de probabilidades para el primer rostro (si está activo)
        if self.show_hud_bars and detecciones_con_emocion:
            self._dibujar_panel_probabilidades(frame, detecciones_con_emocion[0][1], x_pos=15, y_pos=85)

        # 3. Telemetría superior
        if self.show_stats:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Latencia: {latencia_total:.1f}ms | Umbral: {self.classifier.min_confidence:.2f}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Rostros: {len(detecciones_con_emocion)} | [H] HUD  [C] Captura  [T] Umbral  [Q] Salir",
                (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
            )

        # 4. Notificación temporal en pantalla
        if self.notification_text and time.time() < self.notification_expiry:
            (nw, nh), _ = cv2.getTextSize(self.notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            nx = (frame.shape[1] - nw) // 2
            ny = frame.shape[0] - 25

            cv2.rectangle(
                frame,
                (nx - 10, ny - nh - 8),
                (nx + nw + 10, ny + 8),
                (40, 40, 40),
                -1,
            )
            cv2.putText(
                frame,
                self.notification_text,
                (nx, ny),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

    def _guardar_captura_manual(self, frame: np.ndarray) -> str:
        """Guarda la imagen actual en disco bajo demanda explícita del usuario."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre = f"captura_{ts_str}.jpg"
        ruta = self.captures_dir / nombre
        cv2.imwrite(str(ruta), frame)
        return str(ruta)

    def run_webcam(self) -> None:
        """Ejecuta el bucle principal de procesamiento con la cámara web."""
        from test_detector import abrir_camara

        cap = abrir_camara(self.camera_index)
        if cap is None or not cap.isOpened():
            print(f"\n[ERROR] No se pudo acceder a la cámara #{self.camera_index}.")
            print("Verifica permisos en Windows o que otra aplicación no la esté bloqueando.")
            return

        logger = EmotionCSVLogger(source="webcam") if self.enable_logging else None
        nombre_ventana = "Sistema de Reconocimiento Facial y Emociones (Fase 5)"

        print(f"\n[OK] Cámara #{self.camera_index} iniciada con éxito.")
        print("Atajos de teclado:")
        print(" [H]: Alternar panel HUD de probabilidades")
        print(" [C]: Guardar captura de pantalla en data/capturas/")
        print(" [T]: Alternar umbral de confianza (0.35 / 0.45 / 0.60)")
        print(" [S]: Alternar información de telemetría y FPS")
        print(" [Q] o [ESC]: Salir de la aplicación\n")

        frame_count = 0
        tiempos_frame = []
        fallos = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    fallos += 1
                    if fallos >= 30:
                        print("[ERROR] Pérdida de conexión con la cámara web.")
                        break
                    time.sleep(0.1)
                    continue

                fallos = 0
                frame_count += 1
                t0 = time.perf_counter()

                # 1. Detección
                detecciones = self.detector.detect(frame)

                # 2. Clasificación
                resultados: List[Tuple[FaceDetection, EmotionResult]] = []
                for f_idx, det in enumerate(detecciones):
                    crop = FaceDetector.crop_face(frame, det.box, margin=0.15)
                    if crop is not None:
                        em_res = self.classifier.predict(crop)
                        if em_res:
                            resultados.append((det, em_res))
                            self.session_counts[em_res.emotion] = (
                                self.session_counts.get(em_res.emotion, 0) + 1
                            )
                            if logger:
                                logger.log_prediction(frame_id=frame_count, face_id=f_idx, result=em_res)

                dt = (time.perf_counter() - t0) * 1000
                tiempos_frame.append(dt)

                lat_media = np.mean(tiempos_frame[-30:]) if tiempos_frame else dt
                fps = 1000.0 / lat_media if lat_media > 0 else 0

                # 3. Renderizado
                self._renderizar_frame(frame, resultados, fps=fps, latencia_total=lat_media)

                cv2.imshow(nombre_ventana, frame)
                tecla = cv2.waitKey(1) & 0xFF

                if tecla in (ord("q"), ord("Q"), 27):
                    break
                elif tecla in (ord("h"), ord("H")):
                    self.show_hud_bars = not self.show_hud_bars
                    estado = "Visible" if self.show_hud_bars else "Oculto"
                    self._set_notification(f"Panel HUD: {estado}")
                elif tecla in (ord("s"), ord("S")):
                    self.show_stats = not self.show_stats
                    estado = "Visible" if self.show_stats else "Oculto"
                    self._set_notification(f"Telemetría: {estado}")
                elif tecla in (ord("t"), ord("T")):
                    self._current_thresh_idx = (self._current_thresh_idx + 1) % len(self._threshold_levels)
                    nuevo_umbral = self._threshold_levels[self._current_thresh_idx]
                    self.classifier.min_confidence = nuevo_umbral
                    self._set_notification(f"Umbral ajustado a: {nuevo_umbral * 100:.0f}%")
                elif tecla in (ord("c"), ord("C")):
                    ruta_guardada = self._guardar_captura_manual(frame)
                    p_rel = Path(ruta_guardada).name
                    self._set_notification(f"Captura guardada: {p_rel}", duration=3.0)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if logger:
                logger.close()

        # Resumen final de la sesión
        self._imprimir_resumen(len(tiempos_frame), tiempos_frame, logger)

    def run_image(self, image_path: str) -> None:
        """Procesa una imagen estática (JPG/PNG)."""
        from test_classifier import procesar_imagen_estatica

        logger = EmotionCSVLogger(source="imagen") if self.enable_logging else None
        try:
            procesar_imagen_estatica(
                image_path,
                self.detector,
                self.classifier,
                logger=logger,
                mostrar_ventana=True,
            )
        finally:
            if logger:
                logger.close()
                print(f"[OK] Telemetría registrada en: {logger.file_path}")

    def _imprimir_resumen(
        self,
        total_frames: int,
        tiempos: List[float],
        logger: Optional[EmotionCSVLogger],
    ) -> None:
        """Imprime métricas y estadísticas consolidadas al cerrar la sesión."""
        print("\n====================================================")
        print("          RESUMEN DE SESIÓN - FASE 5                ")
        print("====================================================")
        if total_frames > 0 and tiempos:
            lat_media = np.mean(tiempos)
            fps_medio = 1000.0 / lat_media if lat_media > 0 else 0
            print(f" Frames totales procesados: {total_frames}")
            print(f" Latencia promedio por frame: {lat_media:.2f} ms")
            print(f" FPS promedio de la sesión: {fps_medio:.1f} FPS")

            print("\n Distribución de Emociones Registradas:")
            total_emociones = sum(self.session_counts.values())
            if total_emociones > 0:
                for emo, cant in sorted(self.session_counts.items(), key=lambda x: x[1], reverse=True):
                    if cant > 0:
                        pct = (cant / total_emociones) * 100
                        print(f"  - {emo.capitalize():14s}: {cant:4d} veces ({pct:5.1f}%)")

        if logger:
            print(f"\n Log CSV guardado en:\n  {logger.file_path} ({logger.total_logged} registros)")
        print("====================================================\n")

    def close(self) -> None:
        """Libera recursos del detector y clasificador."""
        self.detector.close()
        self.classifier.close()
