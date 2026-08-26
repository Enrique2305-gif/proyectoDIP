import cv2
import mediapipe as mp

from face_utils import (
    FPSCounter,
    SimpleFaceTracker,
    create_face_mesh,
    draw_basic_landmarks,
    get_all_face_landmarks,
    get_face_bbox,
    get_key_points,
    put_label,
    save_frame,
)


MAX_ROSTROS = 5


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Prueba VideoCapture(1).")
        return

    face_mesh = create_face_mesh(max_num_faces=MAX_ROSTROS)
    tracker = SimpleFaceTracker(max_distance=190, max_missed_frames=18)
    fps_counter = FPSCounter()

    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
    show_mesh = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer la cámara.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            faces = get_all_face_landmarks(results)

            face_data = []
            for face_landmarks in faces:
                key_points = get_key_points(face_landmarks, width, height)
                bbox = get_face_bbox(key_points["all"], width, height, padding=18)
                face_data.append((face_landmarks, key_points, bbox))

            face_ids = tracker.update([item[2] for item in face_data])

            put_label(
                frame,
                "FASE 1: detección multirrostro y extracción de landmarks",
                (20, 34),
                0.68,
            )
            put_label(
                frame,
                f"Rostros detectados: {len(face_data)}/{MAX_ROSTROS}",
                (20, 68),
                0.62,
            )

            if not face_data:
                put_label(frame, "No se detectaron rostros", (20, 104), 0.65)

            for (face_landmarks, key_points, bbox), face_id in zip(face_data, face_ids):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)

                if show_mesh:
                    drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mesh_connections,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=(
                            drawing_styles.get_default_face_mesh_tesselation_style()
                        ),
                    )

                draw_basic_landmarks(frame, key_points)
                label_y = y1 - 10 if y1 > 30 else y1 + 24
                put_label(
                    frame,
                    f"Rostro {face_id}",
                    (x1, label_y),
                    0.52,
                    1,
                )

            fps = fps_counter.update()
            put_label(frame, f"FPS: {fps:.1f}", (width - 135, 34), 0.58)
            put_label(
                frame,
                "m = ocultar/mostrar malla | s = captura | q = salir",
                (20, height - 24),
                0.50,
            )

            cv2.imshow("Proyecto - Fase 1 multirrostro", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                save_frame(frame, "fase1_multirrostro")
            if key == ord("m"):
                show_mesh = not show_mesh

    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
