import cv2

from face_utils import put_label, save_frame
from filter_engine import FaceFilterEngine


MAX_ROSTROS = 5
SENSITIVITY_STEP = 0.15


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Prueba VideoCapture(1).")
        return

    engine = FaceFilterEngine(max_num_faces=MAX_ROSTROS)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break

            frame = cv2.flip(frame, 1)
            frame, info = engine.process(frame)
            height, width = frame.shape[:2]

            put_label(
                frame,
                "FASE 3: filtros faciales multirrostro",
                (20, 34),
                0.70,
            )
            put_label(
                frame,
                f"Rostros: {info['face_count']}/{MAX_ROSTROS} | FPS: {info['fps']:.1f} "
                f"| Sensibilidad: {engine.sensitivity:.2f}x",
                (20, 68),
                0.55,
            )

            if info["face_count"] == 0:
                put_label(frame, "No se detectaron rostros", (20, 104), 0.65)
            else:
                put_label(
                    frame,
                    "Cada persona puede cambiar su filtro moviendo la cabeza",
                    (20, height - 54),
                    0.53,
                )

            put_label(
                frame,
                "a/d = filtro global | +/- = sensibilidad | m = landmarks | r = recalibrar | s = captura | q = salir",
                (20, height - 24),
                0.45,
            )

            cv2.imshow("Proyecto - Fase 3 multirrostro", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                save_frame(frame, "fase3_multirrostro")
            if key == ord("r"):
                engine.recalibrate()
                print("Referencias de movimiento reiniciadas.")
            if key == ord("d"):
                engine.next_filter_for_all()
            if key == ord("a"):
                engine.previous_filter_for_all()
            if key == ord("m"):
                engine.show_landmarks = not engine.show_landmarks
            if key in (ord("+"), ord("=")):
                engine.set_sensitivity(engine.sensitivity + SENSITIVITY_STEP)
            if key in (ord("-"), ord("_")):
                engine.set_sensitivity(engine.sensitivity - SENSITIVITY_STEP)

    finally:
        engine.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
