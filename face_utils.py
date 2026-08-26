import math
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Rutas absolutas basadas en la ubicación de este archivo.
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# Índices principales de MediaPipe Face Mesh.
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
MOUTH_CENTER = 13


def create_face_mesh(max_num_faces=5):
    """Crea Face Mesh preparado para detectar varios rostros."""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )


def get_all_face_landmarks(results):
    """Devuelve una lista con todos los rostros detectados."""
    if results is None or not results.multi_face_landmarks:
        return []
    return list(results.multi_face_landmarks)


def get_first_face_landmarks(results):
    """Compatibilidad con fases anteriores: devuelve el primer rostro."""
    faces = get_all_face_landmarks(results)
    return faces[0] if faces else None


def landmark_to_pixel(landmark, width, height):
    """Convierte una coordenada normalizada de MediaPipe a píxeles."""
    x = min(max(int(landmark.x * width), 0), width - 1)
    y = min(max(int(landmark.y * height), 0), height - 1)
    return x, y


def get_key_points(face_landmarks, width, height):
    """Obtiene los puntos clave de un rostro en píxeles."""
    all_points = [
        landmark_to_pixel(point, width, height)
        for point in face_landmarks.landmark
    ]

    return {
        "all": all_points,
        "nose": all_points[NOSE_TIP],
        "forehead": all_points[FOREHEAD],
        "chin": all_points[CHIN],
        "left_eye": all_points[LEFT_EYE_OUTER],
        "right_eye": all_points[RIGHT_EYE_OUTER],
        "mouth": all_points[MOUTH_CENTER],
    }


def get_face_bbox(points, width, height, padding=20):
    """Calcula una caja delimitadora alrededor del rostro."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x1 = max(min(xs) - padding, 0)
    y1 = max(min(ys) - padding, 0)
    x2 = min(max(xs) + padding, width - 1)
    y2 = min(max(ys) + padding, height - 1)
    return x1, y1, x2, y2


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def roll_angle_degrees(left_point, right_point):
    """
    Calcula la inclinación (roll) del rostro en grados a partir de dos
    puntos que deberían estar alineados horizontalmente (p. ej. las
    esquinas externas de los ojos). Se usa para rotar los filtros PNG y
    que acompañen el movimiento de la cabeza en lugar de quedar "pegados"
    en posición horizontal fija.
    """
    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]
    if dx == 0 and dy == 0:
        return 0.0
    return -math.degrees(math.atan2(dy, dx))


def put_label(frame, text, pos, scale=0.75, thickness=2, color=(255, 255, 255)):
    """Escribe texto con sombra para que sea legible sobre el video."""
    x, y = pos
    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_basic_landmarks(frame, key_points, color=(0, 255, 255)):
    """Dibuja y nombra los puntos principales de un rostro."""
    for name, point in key_points.items():
        if name == "all":
            continue
        cv2.circle(frame, point, 5, color, -1)
        cv2.putText(
            frame,
            name,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def save_frame(frame, prefix):
    """Guarda una captura dentro de la carpeta outputs."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUTS_DIR / f"{prefix}_{int(time.time())}.png"
    cv2.imwrite(str(filename), frame)
    print(f"Captura guardada: {filename}")
    return str(filename)


class FPSCounter:
    """Calcula un FPS suavizado para mostrar el rendimiento del prototipo."""

    def __init__(self, smoothing=0.90):
        self.last_time = time.perf_counter()
        self.fps = 0.0
        self.smoothing = smoothing

    def update(self):
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            if self.fps == 0.0:
                self.fps = instant_fps
            else:
                self.fps = (
                    self.smoothing * self.fps
                    + (1.0 - self.smoothing) * instant_fps
                )
        return self.fps


class SimpleFaceTracker:
    """
    Asigna un identificador temporal a cada rostro usando la cercanía entre
    centros de cajas. Es suficiente para una demo y evita depender del orden
    cambiante con el que MediaPipe devuelve los rostros.

    Mejora respecto a la versión inicial: cada pista guarda una velocidad
    suavizada (desplazamiento por fotograma) y el emparejamiento se hace
    contra la posición PREDICHA (centro + velocidad), no contra la última
    posición conocida. Esto reduce los cambios de identificador cuando un
    rostro se mueve rápido o cuando dos personas se cruzan, porque el
    sistema "anticipa" hacia dónde va cada rostro en lugar de asumir que
    se quedó quieto.
    """

    def __init__(self, max_distance=180, max_missed_frames=18, velocity_smoothing=0.6):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.velocity_smoothing = velocity_smoothing
        self.next_id = 1
        self.tracks = {}

    def _predicted_center(self, track):
        center = track["center"]
        velocity = track["velocity"]
        return (center[0] + velocity[0], center[1] + velocity[1])

    def update(self, bboxes):
        centers = [bbox_center(bbox) for bbox in bboxes]

        for track in self.tracks.values():
            track["missed"] += 1

        predicted = {
            track_id: self._predicted_center(track)
            for track_id, track in self.tracks.items()
        }

        assigned_track_ids = [None] * len(centers)
        used_tracks = set()

        # Se emparejan primero las detecciones más cercanas a una posición
        # predicha (no a la última posición real) para tolerar movimientos
        # rápidos sin perder el identificador del rostro.
        possible_matches = []
        for detection_index, center in enumerate(centers):
            for track_id, predicted_center in predicted.items():
                match_distance = distance(center, predicted_center)
                possible_matches.append((match_distance, detection_index, track_id))

        possible_matches.sort(key=lambda item: item[0])

        for match_distance, detection_index, track_id in possible_matches:
            if assigned_track_ids[detection_index] is not None:
                continue
            if track_id in used_tracks:
                continue
            if match_distance > self.max_distance:
                continue

            assigned_track_ids[detection_index] = track_id
            used_tracks.add(track_id)

            old_center = self.tracks[track_id]["center"]
            old_velocity = self.tracks[track_id]["velocity"]
            new_center = centers[detection_index]

            instant_velocity = (
                new_center[0] - old_center[0],
                new_center[1] - old_center[1],
            )

            # Suavizado exponencial: evita que un salto puntual (ruido de
            # detección) dispare una predicción exagerada en el siguiente
            # fotograma.
            smoothed_velocity = (
                self.velocity_smoothing * old_velocity[0]
                + (1.0 - self.velocity_smoothing) * instant_velocity[0],
                self.velocity_smoothing * old_velocity[1]
                + (1.0 - self.velocity_smoothing) * instant_velocity[1],
            )

            self.tracks[track_id] = {
                "center": new_center,
                "velocity": smoothed_velocity,
                "missed": 0,
            }

        # Las detecciones sin coincidencia reciben un nuevo ID.
        for detection_index, center in enumerate(centers):
            if assigned_track_ids[detection_index] is None:
                track_id = self.next_id
                self.next_id += 1
                assigned_track_ids[detection_index] = track_id
                self.tracks[track_id] = {
                    "center": center,
                    "velocity": (0.0, 0.0),
                    "missed": 0,
                }

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed_frames
        ]
        for track_id in expired:
            del self.tracks[track_id]

        return assigned_track_ids

    @property
    def active_ids(self):
        return set(self.tracks.keys())
