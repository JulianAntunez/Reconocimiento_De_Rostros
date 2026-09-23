"""
Script de Prueba Individual para la Fase 2: Detección de Rostros
Permite probar:
1. Webcam en vivo con recuadros y puntos clave dibujados en pantalla.
2. Modo no interactivo / línea de comandos (toma frames e imprime estadísticas y latencia en ms).
3. Imágenes estáticas (JPG/PNG).
"""

from typing import Optional
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Asegurar importación de módulos locales
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG
from core import FaceDetector


def procesar_imagen_estatica(ruta_imagen: str, detector: FaceDetector, mostrar_ventana: bool = True) -> None:
    """Prueba la detección en un archivo de imagen (JPG/PNG)."""
    p = Path(ruta_imagen)
    if not p.is_file():
        print(f"[ERROR] El archivo de imagen no existe: {ruta_imagen}")
        return

    # Leer imagen tolerando caracteres especiales en la ruta
    frame = cv2.imread(str(p))
    if frame is None or frame.size == 0:
        print(f"[ERROR] No se pudo decodificar la imagen. Formato no soportado o archivo corrupto: {ruta_imagen}")
        return

    t_inicio = time.perf_counter()
    detecciones = detector.detect(frame)
    latencia_ms = (time.perf_counter() - t_inicio) * 1000

    print(f"\n--- Resultado para {p.name} ---")
    print(f" Dimensiones: {frame.shape[1]}x{frame.shape[0]}")
    print(f" Rostros detectados: {len(detecciones)}")
    print(f" Tiempo de inferencia: {latencia_ms:.2f} ms")

    if not detecciones:
        print(" [AVISO] Ningún rostro detectado en la imagen.")
        return

    for idx, det in enumerate(detecciones, start=1):
        x, y, w, h = det.box
        print(f"  Rostro #{idx}: Box (x={x}, y={y}, w={w}, h={h}) | Confianza: {det.confidence * 100:.1f}%")
        # Dibujar recuadro en verde
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Rostro {idx}: {det.confidence * 100:.1f}%",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if mostrar_ventana:
        print("\nPresiona cualquier tecla en la ventana de imagen para cerrarla...")
        cv2.imshow("Prueba de Deteccion - Imagen", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def abrir_camara(indice_camara: int = 0) -> Optional[cv2.VideoCapture]:
    """Abre la cámara web con el backend nativo del sistema de forma segura."""
    cap = cv2.VideoCapture(indice_camara)
    if not cap.isOpened():
        cap = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)
    return cap if cap.isOpened() else None


def procesar_webcam(
    detector: FaceDetector,
    indice_camara: int = 0,
    modo_headless: bool = False,
    max_frames_headless: int = 15,
) -> None:
    """Prueba la detección en tiempo real desde la webcam."""
    cap = abrir_camara(indice_camara)

    if cap is None or not cap.isOpened():
        print(f"\n[ERROR] No se pudo abrir la cámara en el índice {indice_camara}.")
        print("Causas posibles:")
        print(" 1. Otra aplicación tiene la cámara abierta (app 'Cámara de Windows', Teams, Zoom, Meet).")
        print(" 2. Permisos de cámara bloqueados en la configuración de Windows.")
        print(" 3. Si usas cámara externa USB, prueba con '--cam 1'.")
        return

    print(f"\n[OK] Cámara #{indice_camara} conectada.")
    nombre_ventana = "Fase 2: Prueba de Deteccion (MediaPipe)"

    if not modo_headless:
        cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
        print("Iniciando ventana en vivo. Presiona 'q' o 'ESC' para salir.")
    else:
        print(f"Modo no interactivo: capturando {max_frames_headless} frames para medir rendimiento...")

    frame_count = 0
    fallos_consecutivos = 0
    tiempos = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                fallos_consecutivos += 1
                if fallos_consecutivos == 1:
                    print("Esperando señal de video de la cámara (asegúrate de cerrar otras apps de cámara)...")
                if fallos_consecutivos >= 30:
                    print("\n[ERROR] No se pudo leer video de la cámara. Verifica que ninguna otra app la esté utilizando.")
                    break
                time.sleep(0.1)
                continue

            fallos_consecutivos = 0

            t0 = time.perf_counter()
            detecciones = detector.detect(frame)
            dt = (time.perf_counter() - t0) * 1000
            tiempos.append(dt)
            frame_count += 1

            if modo_headless:
                print(f" Frame {frame_count:02d}: {len(detecciones)} rostro(s) detectado(s) en {dt:.2f} ms")
                if frame_count >= max_frames_headless:
                    break
            else:
                # Alerta si el sensor entrega una imagen completamente negra
                brillo_medio = np.mean(frame)
                if brillo_medio < 10.0:
                    cv2.putText(
                        frame,
                        "CUIDADO: Sensor oscuro. Revisa la tapa de la webcam.",
                        (15, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

                # Dibujar detecciones
                for idx, det in enumerate(detecciones, start=1):
                    x, y, w, h = det.box
                    # Recuadro verde
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    etiqueta = f"Cara #{idx}: {det.confidence * 100:.1f}%"
                    cv2.putText(
                        frame,
                        etiqueta,
                        (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    # Puntos clave si están disponibles
                    if det.landmarks:
                        for _, (lx, ly) in det.landmarks.items():
                            cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)

                # Información en pantalla
                latencia_media = np.mean(tiempos[-30:]) if tiempos else dt
                fps_aprox = 1000.0 / latencia_media if latencia_media > 0 else 0
                cv2.putText(
                    frame,
                    f"Inferencia: {latencia_media:.1f}ms (~{fps_aprox:.0f} FPS CPU)",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Rostros: {len(detecciones)} (Presiona 'q' para salir)",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow(nombre_ventana, frame)
                tecla = cv2.waitKey(10) & 0xFF
                if tecla in (ord("q"), ord("Q"), 27):  # 'q' o ESC
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if tiempos:
        print("\n--- Estadísticas de Rendimiento en CPU ---")
        print(f" Total frames procesados: {len(tiempos)}")
        print(f" Latencia promedio por frame: {np.mean(tiempos):.2f} ms")
        print(f" FPS teóricos máximos solo de detección: {1000.0 / np.mean(tiempos):.1f} FPS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba individual del detector de rostros (MediaPipe)")
    parser.add_argument("--image", type=str, default=None, help="Ruta a una imagen JPG o PNG para probar")
    parser.add_argument("--cam", type=int, default=DEFAULT_CONFIG.camera_index, help="Índice de la cámara")
    parser.add_argument("--headless", action="store_true", help="Modo consola sin abrir ventana gráfica")
    args = parser.parse_args()

    detector = FaceDetector(min_confidence=DEFAULT_CONFIG.face_detection_confidence)

    try:
        if args.image:
            procesar_imagen_estatica(args.image, detector, mostrar_ventana=not args.headless)
        else:
            procesar_webcam(detector, indice_camara=args.cam, modo_headless=args.headless)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
