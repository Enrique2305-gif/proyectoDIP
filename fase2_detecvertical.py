import cv2

from face_utils import (
    FPSCounter,
    create_face_mesh,
    get_first_face_landmarks,
    get_face_bbox,
    get_key_points,
    put_label,
    save_frame,
)
from motion_detector import VerticalMotionDetector


def draw_arrow(frame, direction, center):
    x, y = center
    if direction == "ARRIBA":
        cv2.arrowedLine(frame, (x, y + 70), (x, y - 70), (0, 255, 0), 8, tipLength=0.35)
    elif direction == "ABAJO":
        cv2.arrowedLine(frame, (x, y - 70), (x, y + 70), (0, 0, 255), 8, tipLength=0.35)
    else:
        cv2.circle(frame, (x, y), 35, (0, 255, 255), 5)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Prueba VideoCapture(1).")
        return

    face_mesh = create_face_mesh(max_num_faces=1)
    detector = VerticalMotionDetector(
        threshold_ratio=0.11,
        cooldown_seconds=0.85,
        smoothing_alpha=0.35,
    )
    fps_counter = FPSCounter()
    sensitivity_step = 0.15

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            face_landmarks = get_first_face_landmarks(results)

            put_label(frame, "FASE 2: detección del movimiento vertical", (20, 35), 0.7)
            put_label(
                frame,
                f"Sensibilidad: {detector.sensitivity:.2f}x  "
                "(+/- para ajustar)",
                (20, height - 55),
                0.55,
            )
            put_label(frame, "r = recalibrar | s = captura | q = salir", (20, height - 25), 0.55)

            if face_landmarks is None:
                detector.reset()
                put_label(frame, "Rostro no detectado", (20, 75), 0.7)
            else:
                key_points = get_key_points(face_landmarks, width, height)
                x1, y1, x2, y2 = get_face_bbox(key_points["all"], width, height)
                face_height = y2 - y1
                nose = key_points["nose"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, nose, 8, (0, 255, 255), -1)
                cv2.line(frame, (0, nose[1]), (width, nose[1]), (255, 255, 0), 1)

                direction, delta, threshold = detector.update(nose[1], face_height)
                draw_arrow(frame, direction, (width - 120, 160))

                put_label(frame, f"Nariz suavizada: y={nose[1]}", (20, 75), 0.6)
                put_label(frame, f"Delta vertical: {delta:.1f} px", (20, 105), 0.6)
                put_label(frame, f"Umbral dinámico: {threshold} px", (20, 135), 0.9)
                put_label(frame, f"Movimiento: {direction}", (20, 170), 0.8)

            fps = fps_counter.update()
            put_label(frame, f"FPS: {fps:.1f}", (width - 135, 35), 0.58)

            cv2.imshow("Proyecto - Fase 2", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_frame(frame, "fase2_movimiento")
            if key == ord("r"):
                detector.reset()
                print("Referencia reiniciada.")
            if key in (ord("+"), ord("=")):
                detector.set_sensitivity(detector.sensitivity + sensitivity_step)
            if key in (ord("-"), ord("_")):
                detector.set_sensitivity(detector.sensitivity - sensitivity_step)

    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
