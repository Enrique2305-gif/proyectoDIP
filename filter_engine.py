import subprocess
import sys

import cv2
import mediapipe as mp
import numpy as np

from face_utils import (
    BASE_DIR,
    FPSCounter,
    SimpleFaceTracker,
    bbox_area,
    create_face_mesh,
    distance,
    get_all_face_landmarks,
    get_face_bbox,
    get_key_points,
    midpoint,
    put_label,
    roll_angle_degrees,
)
from motion_detector import VerticalMotionDetector

# Ancho máximo (en píxeles) que se envía a MediaPipe para la detección.
# Si el fotograma de la cámara es más ancho, se procesa una copia reducida
# (los landmarks son coordenadas normalizadas 0-1, así que siguen siendo
# válidos sobre el fotograma original). Reduce notablemente el costo de
# detectar varios rostros a la vez sin afectar la precisión visible.
DETECTION_MAX_WIDTH = 720


# ============================================================
# ADMINISTRACIÓN DE FILTROS
# ============================================================

class FilterManager:
    def __init__(self):
        self.filters = [
            {
                "name": "Gafas",
                "file": "assets/glasses.png",
                "type": "glasses",
            },
            {
                "name": "Bigote",
                "file": "assets/mustache.png",
                "type": "mustache",
            },
            {
                "name": "Corona",
                "file": "assets/crown.png",
                "type": "crown",
            },
            {
                "name": "Orejas de gato",
                "file": "assets/cat_ears.png",
                "type": "cat",
            },
            {
                "name": "Lentes de sol",
                "file": "assets/sunglasses.png",
                "type": "sunglasses",
            },
            {
                "name": "Sonrisa de Guasón",
                "file": None,
                "type": "joker_smile",
            },
            {
                "name": "Ojos grandes",
                "file": None,
                "type": "big_eyes",
            },
            {
                "name": "Corazones",
                "file": None,
                "type": "heart_eyes",
            },
        ]

        self.images = []
        self.load_filters()

    def load_filters(self):
        """
        Carga los filtros PNG.

        Los filtros procedurales, como la sonrisa de Guasón
        y los ojos grandes, no utilizan imágenes.
        """
        missing = []

        for filter_info in self.filters:
            file_name = filter_info.get("file")

            if not file_name:
                continue

            path = BASE_DIR / file_name

            if not path.exists():
                missing.append(path)

        if missing:
            generator = BASE_DIR / "generate_assets.py"

            print(
                "Faltan filtros PNG. "
                "Se generarán recursos de ejemplo."
            )

            subprocess.run(
                [sys.executable, str(generator)],
                cwd=BASE_DIR,
                check=False,
            )

        self.images.clear()

        for filter_info in self.filters:
            file_name = filter_info.get("file")

            if not file_name:
                self.images.append(None)
                continue

            path = BASE_DIR / file_name

            image = cv2.imread(
                str(path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                raise FileNotFoundError(
                    f"No se pudo cargar el filtro: {path}"
                )

            self.images.append(image)

    def get(self, index):
        normalized_index = index % len(self.filters)

        return (
            self.filters[normalized_index],
            self.images[normalized_index],
        )

    def normalize(self, index):
        return index % len(self.filters)

    @property
    def count(self):
        return len(self.filters)

    @property
    def names(self):
        return [
            filter_info["name"]
            for filter_info in self.filters
        ]


# ============================================================
# SUPERPOSICIÓN DE FILTROS PNG
# ============================================================

def rotate_rgba_image(image, angle_degrees):
    """
    Rota una imagen RGBA alrededor de su propio centro, conservando el
    canal alfa (las zonas que quedan fuera se rellenan como transparentes).
    Se usa para que los filtros PNG (gafas, bigote, corona, orejas, lentes
    de sol) acompañen la inclinación lateral (roll) de la cabeza en lugar
    de quedar siempre perfectamente horizontales.
    """
    if abs(angle_degrees) < 1.0:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def overlay_png(frame, overlay, x, y, width, height, angle_degrees=0.0):
    """
    Superpone una imagen PNG con transparencia sobre el video.

    Si se indica angle_degrees, la imagen se rota antes de superponerse
    (usado para que los filtros sigan la inclinación de la cabeza).
    """
    if overlay is None or width <= 0 or height <= 0:
        return frame

    interpolation = (
        cv2.INTER_AREA
        if width < overlay.shape[1]
        else cv2.INTER_LINEAR
    )

    overlay = cv2.resize(
        overlay,
        (width, height),
        interpolation=interpolation,
    )

    if angle_degrees:
        overlay = rotate_rgba_image(overlay, angle_degrees)

    frame_height, frame_width = frame.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + width, frame_width)
    y2 = min(y + height, frame_height)

    if x1 >= x2 or y1 >= y2:
        return frame

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[
        overlay_y1:overlay_y2,
        overlay_x1:overlay_x2,
    ]

    if overlay_crop.ndim != 3:
        return frame

    if overlay_crop.shape[2] == 4:
        alpha = (
            overlay_crop[:, :, 3:4].astype(np.float32)
            / 255.0
        )

        color = overlay_crop[
            :,
            :,
            :3,
        ].astype(np.float32)

    else:
        alpha = np.ones(
            (*overlay_crop.shape[:2], 1),
            dtype=np.float32,
        )

        color = overlay_crop[
            :,
            :,
            :3,
        ].astype(np.float32)

    roi = frame[
        y1:y2,
        x1:x2,
    ].astype(np.float32)

    blended = (
        alpha * color
        + (1.0 - alpha) * roi
    )

    frame[
        y1:y2,
        x1:x2,
    ] = blended.astype(np.uint8)

    return frame


# ============================================================
# FILTRO SONRISA DE GUASÓN
# ============================================================

def quadratic_bezier_curve(
    point_1,
    control_point,
    point_2,
    steps=60,
):
    """
    Genera una curva de Bézier cuadrática.
    """
    t = np.linspace(
        0.0,
        1.0,
        steps,
        dtype=np.float32,
    )

    point_1 = np.asarray(
        point_1,
        dtype=np.float32,
    )

    control_point = np.asarray(
        control_point,
        dtype=np.float32,
    )

    point_2 = np.asarray(
        point_2,
        dtype=np.float32,
    )

    curve = (
        ((1.0 - t) ** 2)[:, None] * point_1
        + (
            2.0
            * (1.0 - t)
            * t
        )[:, None] * control_point
        + (t ** 2)[:, None] * point_2
    )

    return (
        curve
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )


def warp_local_region(frame, controls, roi):
    """
    Deforma localmente una región de la imagen.

    Cada control contiene:

        punto original,
        punto destino,
        radio de influencia.
    """
    frame_height, frame_width = frame.shape[:2]

    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    roi_x1 = max(
        0,
        min(
            int(roi_x1),
            frame_width - 1,
        ),
    )

    roi_y1 = max(
        0,
        min(
            int(roi_y1),
            frame_height - 1,
        ),
    )

    roi_x2 = max(
        0,
        min(
            int(roi_x2),
            frame_width,
        ),
    )

    roi_y2 = max(
        0,
        min(
            int(roi_y2),
            frame_height,
        ),
    )

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return frame

    original_region = frame[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ].copy()

    region_height, region_width = (
        original_region.shape[:2]
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(
            region_width,
            dtype=np.float32,
        ),
        np.arange(
            region_height,
            dtype=np.float32,
        ),
    )

    map_x = grid_x.copy()
    map_y = grid_y.copy()

    for source, target, radius in controls:
        source_x = float(
            source[0] - roi_x1
        )

        source_y = float(
            source[1] - roi_y1
        )

        target_x = float(
            target[0] - roi_x1
        )

        target_y = float(
            target[1] - roi_y1
        )

        displacement_x = (
            target_x - source_x
        )

        displacement_y = (
            target_y - source_y
        )

        radius = max(
            float(radius),
            1.0,
        )

        sigma = max(
            radius * 0.48,
            1.0,
        )

        distance_squared = (
            (grid_x - target_x) ** 2
            + (grid_y - target_y) ** 2
        )

        influence = np.exp(
            -distance_squared
            / (2.0 * sigma ** 2)
        )

        influence[
            distance_squared > radius ** 2
        ] = 0.0

        map_x -= (
            displacement_x
            * influence
        )

        map_y -= (
            displacement_y
            * influence
        )

    map_x = np.clip(
        map_x,
        0,
        max(region_width - 1, 0),
    ).astype(np.float32)

    map_y = np.clip(
        map_y,
        0,
        max(region_height - 1, 0),
    ).astype(np.float32)

    warped_region = cv2.remap(
        original_region,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    frame[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ] = warped_region

    return frame


def apply_joker_makeup(
    frame,
    left_target,
    right_target,
    upper_lip,
    lower_lip,
    face_width,
    face_height,
):
    """
    Dibuja el maquillaje rojo sobre la sonrisa deformada.
    """
    center_x = int(
        (
            left_target[0]
            + right_target[0]
        )
        / 2
    )

    upper_control = (
        center_x,
        int(
            upper_lip[1]
            + face_height * 0.025
        ),
    )

    lower_control = (
        center_x,
        int(
            lower_lip[1]
            + face_height * 0.060
        ),
    )

    upper_curve = quadratic_bezier_curve(
        left_target,
        upper_control,
        right_target,
    )

    lower_curve = quadratic_bezier_curve(
        left_target,
        lower_control,
        right_target,
    )

    line_thickness = max(
        3,
        int(face_width * 0.021),
    )

    dark_red = (20, 20, 95)
    bright_red = (25, 25, 220)

    cv2.polylines(
        frame,
        [upper_curve],
        False,
        dark_red,
        line_thickness + 4,
        cv2.LINE_AA,
    )

    cv2.polylines(
        frame,
        [lower_curve],
        False,
        dark_red,
        line_thickness + 4,
        cv2.LINE_AA,
    )

    cv2.polylines(
        frame,
        [upper_curve],
        False,
        bright_red,
        line_thickness,
        cv2.LINE_AA,
    )

    cv2.polylines(
        frame,
        [lower_curve],
        False,
        bright_red,
        line_thickness,
        cv2.LINE_AA,
    )

    extension = max(
        5,
        int(face_width * 0.050),
    )

    left_extension = (
        left_target[0] - extension,
        left_target[1] - extension // 3,
    )

    right_extension = (
        right_target[0] + extension,
        right_target[1] - extension // 3,
    )

    cv2.line(
        frame,
        left_target,
        left_extension,
        bright_red,
        line_thickness,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        right_target,
        right_extension,
        bright_red,
        line_thickness,
        cv2.LINE_AA,
    )

    return frame


def apply_joker_smile(frame, key_points, bbox):
    """
    Estira las comisuras de los labios hacia los lados
    y hacia arriba.

    Landmarks utilizados:

        61  = primera comisura
        291 = segunda comisura
        13  = labio superior
        14  = labio inferior
    """
    all_points = key_points["all"]

    mouth_corner_1 = all_points[61]
    mouth_corner_2 = all_points[291]
    upper_lip = all_points[13]
    lower_lip = all_points[14]

    if mouth_corner_1[0] <= mouth_corner_2[0]:
        left_corner = mouth_corner_1
        right_corner = mouth_corner_2
    else:
        left_corner = mouth_corner_2
        right_corner = mouth_corner_1

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

    face_width = max(
        1,
        bbox_x2 - bbox_x1,
    )

    face_height = max(
        1,
        bbox_y2 - bbox_y1,
    )

    frame_height, frame_width = frame.shape[:2]

    horizontal_stretch = max(
        10,
        int(face_width * 0.15),
    )

    vertical_lift = max(
        5,
        int(face_height * 0.065),
    )

    left_target = (
        max(
            0,
            left_corner[0]
            - horizontal_stretch,
        ),
        max(
            0,
            left_corner[1]
            - vertical_lift,
        ),
    )

    right_target = (
        min(
            frame_width - 1,
            right_corner[0]
            + horizontal_stretch,
        ),
        max(
            0,
            right_corner[1]
            - vertical_lift,
        ),
    )

    lower_lip_target = (
        lower_lip[0],
        min(
            frame_height - 1,
            lower_lip[1]
            + int(face_height * 0.025),
        ),
    )

    corner_radius = max(
        22,
        int(face_width * 0.28),
    )

    lower_lip_radius = max(
        18,
        int(face_width * 0.18),
    )

    roi_margin_x = int(
        face_width * 0.24
    )

    roi_margin_y = int(
        face_height * 0.18
    )

    roi = (
        left_target[0] - roi_margin_x,
        min(
            left_target[1],
            upper_lip[1],
        ) - roi_margin_y,
        right_target[0] + roi_margin_x,
        max(
            lower_lip_target[1],
            lower_lip[1],
        ) + roi_margin_y,
    )

    controls = [
        (
            left_corner,
            left_target,
            corner_radius,
        ),
        (
            right_corner,
            right_target,
            corner_radius,
        ),
        (
            lower_lip,
            lower_lip_target,
            lower_lip_radius,
        ),
    ]

    warp_local_region(
        frame,
        controls,
        roi,
    )

    apply_joker_makeup(
        frame,
        left_target,
        right_target,
        upper_lip,
        lower_lip_target,
        face_width,
        face_height,
    )

    return frame


# ============================================================
# FILTRO OJOS GRANDES
# ============================================================

def magnify_elliptical_region(
    frame,
    center,
    radius_x,
    radius_y,
    zoom=5,
):
    """
    Amplía una región elíptica de manera progresiva.

    La ampliación es máxima en el centro y disminuye
    hasta desaparecer en los bordes.
    """
    frame_height, frame_width = frame.shape[:2]

    center_x, center_y = center

    radius_x = max(
        int(radius_x),
        1,
    )

    radius_y = max(
        int(radius_y),
        1,
    )

    margin_x = int(
        radius_x * 1.15
    )

    margin_y = int(
        radius_y * 1.15
    )

    x1 = max(
        0,
        center_x - margin_x,
    )

    y1 = max(
        0,
        center_y - margin_y,
    )

    x2 = min(
        frame_width,
        center_x + margin_x + 1,
    )

    y2 = min(
        frame_height,
        center_y + margin_y + 1,
    )

    if x2 <= x1 or y2 <= y1:
        return frame

    original_region = frame[
        y1:y2,
        x1:x2,
    ].copy()

    region_height, region_width = (
        original_region.shape[:2]
    )

    grid_x, grid_y = np.meshgrid(
        np.arange(
            region_width,
            dtype=np.float32,
        ),
        np.arange(
            region_height,
            dtype=np.float32,
        ),
    )

    global_x = grid_x + x1
    global_y = grid_y + y1

    difference_x = (
        global_x - float(center_x)
    )

    difference_y = (
        global_y - float(center_y)
    )

    normalized_distance = np.sqrt(
        (
            difference_x
            / float(radius_x)
        ) ** 2
        + (
            difference_y
            / float(radius_y)
        ) ** 2
    )

    inside_region = (
        normalized_distance <= 1.0
    )

    normalized_clipped = np.clip(
        normalized_distance,
        0.0,
        1.0,
    )

    influence = (
        1.0
        - normalized_clipped ** 2
    ) ** 2

    local_zoom = (
        1.0
        + (zoom - 1.0)
        * influence
    )

    source_global_x = (
        float(center_x)
        + difference_x / local_zoom
    )

    source_global_y = (
        float(center_y)
        + difference_y / local_zoom
    )

    map_x = source_global_x - x1
    map_y = source_global_y - y1

    map_x = np.where(
        inside_region,
        map_x,
        grid_x,
    )

    map_y = np.where(
        inside_region,
        map_y,
        grid_y,
    )

    map_x = np.clip(
        map_x,
        0,
        region_width - 1,
    ).astype(np.float32)

    map_y = np.clip(
        map_y,
        0,
        region_height - 1,
    ).astype(np.float32)

    magnified_region = cv2.remap(
        original_region,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    frame[
        y1:y2,
        x1:x2,
    ] = magnified_region

    return frame


def get_eye_region(
    all_points,
    outer_corner_index,
    inner_corner_index,
    upper_eyelid_index,
    lower_eyelid_index,
):
    """
    Calcula el centro y el tamaño aproximado de un ojo.
    """
    outer_corner = all_points[
        outer_corner_index
    ]

    inner_corner = all_points[
        inner_corner_index
    ]

    upper_eyelid = all_points[
        upper_eyelid_index
    ]

    lower_eyelid = all_points[
        lower_eyelid_index
    ]

    center_x = int(
        (
            outer_corner[0]
            + inner_corner[0]
            + upper_eyelid[0]
            + lower_eyelid[0]
        )
        / 4
    )

    center_y = int(
        (
            outer_corner[1]
            + inner_corner[1]
            + upper_eyelid[1]
            + lower_eyelid[1]
        )
        / 4
    )

    eye_width = distance(
        outer_corner,
        inner_corner,
    )

    eye_height = distance(
        upper_eyelid,
        lower_eyelid,
    )

    radius_x = max(
        18,
        int(eye_width * 0.90),
    )

    radius_y = max(
        14,
        int(
            max(
                eye_height * 2.10,
                eye_width * 0.48,
            )
        ),
    )

    return (
        (center_x, center_y),
        radius_x,
        radius_y,
    )


def apply_big_eyes(frame, key_points, bbox):
    """
    Agranda ambos ojos utilizando landmarks faciales.

    Primer ojo:
        33  = esquina exterior
        133 = esquina interior
        159 = párpado superior
        145 = párpado inferior

    Segundo ojo:
        263 = esquina exterior
        362 = esquina interior
        386 = párpado superior
        374 = párpado inferior
    """
    del bbox

    all_points = key_points["all"]

    (
        first_eye_center,
        first_radius_x,
        first_radius_y,
    ) = get_eye_region(
        all_points,
        outer_corner_index=33,
        inner_corner_index=133,
        upper_eyelid_index=159,
        lower_eyelid_index=145,
    )

    (
        second_eye_center,
        second_radius_x,
        second_radius_y,
    ) = get_eye_region(
        all_points,
        outer_corner_index=263,
        inner_corner_index=362,
        upper_eyelid_index=386,
        lower_eyelid_index=374,
    )

    magnify_elliptical_region(
        frame,
        first_eye_center,
        first_radius_x,
        first_radius_y,
        zoom=5,
    )

    magnify_elliptical_region(
        frame,
        second_eye_center,
        second_radius_x,
        second_radius_y,
        zoom=5,
    )

    return frame


# ============================================================
# FILTRO CORAZONES EN LOS OJOS
# ============================================================

def draw_heart(frame, center, size, fill_color, outline_color):
    """
    Dibuja un corazón relleno usando una curva paramétrica clásica,
    escalado y centrado en 'center'.
    """
    t = np.linspace(0.0, 2.0 * np.pi, 60, dtype=np.float32)

    heart_x = 16.0 * np.sin(t) ** 3
    heart_y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - np.cos(4.0 * t)
    )

    scale = size / 16.0
    points_x = center[0] + heart_x * scale
    # El eje Y de la fórmula crece hacia arriba; en la imagen crece hacia
    # abajo, así que se invierte para que el corazón quede "derecho".
    points_y = center[1] - heart_y * scale

    points = (
        np.stack([points_x, points_y], axis=1)
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )

    cv2.fillPoly(frame, [points], fill_color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [points], True, outline_color, 2, cv2.LINE_AA)


def apply_heart_eyes(frame, key_points):
    """
    Dibuja un corazón sobre cada ojo, con el tamaño ajustado al ancho
    de cada ojo (mismos landmarks que usa el filtro "Ojos grandes").
    """
    all_points = key_points["all"]

    (first_eye_center, first_radius_x, _) = get_eye_region(
        all_points,
        outer_corner_index=33,
        inner_corner_index=133,
        upper_eyelid_index=159,
        lower_eyelid_index=145,
    )

    (second_eye_center, second_radius_x, _) = get_eye_region(
        all_points,
        outer_corner_index=263,
        inner_corner_index=362,
        upper_eyelid_index=386,
        lower_eyelid_index=374,
    )

    fill_color = (70, 40, 235)
    outline_color = (255, 255, 255)

    draw_heart(
        frame,
        first_eye_center,
        max(16, int(first_radius_x * 0.95)),
        fill_color,
        outline_color,
    )

    draw_heart(
        frame,
        second_eye_center,
        max(16, int(second_radius_x * 0.95)),
        fill_color,
        outline_color,
    )

    return frame


# ============================================================
# POSICIÓN DE FILTROS PNG
# ============================================================

def compute_filter_position(
    filter_type,
    filter_img,
    key_points,
    bbox,
):
    """
    Calcula la posición y el tamaño de los filtros PNG.
    """
    if filter_img is None:
        return 0, 0, 0, 0

    x1, y1, x2, y2 = bbox

    face_width = max(
        1,
        x2 - x1,
    )

    left_eye = key_points["left_eye"]
    right_eye = key_points["right_eye"]
    nose = key_points["nose"]
    mouth = key_points["mouth"]
    forehead = key_points["forehead"]

    image_height, image_width = (
        filter_img.shape[:2]
    )

    aspect_ratio = (
        image_height
        / max(1, image_width)
    )

    if filter_type in ("glasses", "sunglasses"):
        eye_center = midpoint(
            left_eye,
            right_eye,
        )

        eye_distance = distance(
            left_eye,
            right_eye,
        )

        width = int(
            eye_distance * 2.15
        )

        height = int(
            width * aspect_ratio
        )

        x = int(
            eye_center[0]
            - width / 2
        )

        y = int(
            eye_center[1]
            - height * 0.48
        )

    elif filter_type == "mustache":
        center_x = nose[0]

        center_y = int(
            (
                nose[1]
                + mouth[1]
            )
            / 2
        )

        width = int(
            face_width * 0.56
        )

        height = int(
            width * aspect_ratio
        )

        x = int(
            center_x
            - width / 2
        )

        y = int(
            center_y
            - height * 0.36
        )

    elif filter_type == "crown":
        width = int(
            face_width * 1.18
        )

        height = int(
            width * aspect_ratio
        )

        x = int(
            (x1 + x2) / 2
            - width / 2
        )

        y = int(
            forehead[1]
            - height * 0.90
        )

    elif filter_type == "cat":
        width = int(
            face_width * 1.27
        )

        height = int(
            width * aspect_ratio
        )

        x = int(
            (x1 + x2) / 2
            - width / 2
        )

        y = int(
            forehead[1]
            - height * 0.68
        )

    else:
        width = face_width

        height = int(
            width * aspect_ratio
        )

        x = int(
            (x1 + x2) / 2
            - width / 2
        )

        y = int(
            (y1 + y2) / 2
            - height / 2
        )

    return x, y, width, height


# ============================================================
# MOTOR PRINCIPAL
# ============================================================

class FaceFilterEngine:
    """
    Motor principal de filtros faciales.

    Permite:

    - Detectar hasta cinco rostros.
    - Mantener un filtro independiente por rostro.
    - Cambiar filtros mediante movimiento vertical.
    - Aplicar imágenes PNG.
    - Aplicar la sonrisa de Guasón.
    - Aplicar ojos grandes.
    """

    def __init__(self, max_num_faces=5):
        self.max_num_faces = max_num_faces

        self.face_mesh = create_face_mesh(
            max_num_faces=max_num_faces
        )

        self.manager = FilterManager()

        self.tracker = SimpleFaceTracker(
            max_distance=190,
            max_missed_frames=18,
        )

        self.motion_detectors = {}
        self.filter_indices = {}

        self.default_filter_index = 0
        self.show_landmarks = False
        self.auto_change = True
        self.sensitivity = 1.0

        self.fps_counter = FPSCounter()

        self.drawing = (
            mp.solutions.drawing_utils
        )

        self.drawing_styles = (
            mp.solutions.drawing_styles
        )

        self.mesh_connections = (
            mp.solutions
            .face_mesh
            .FACEMESH_TESSELATION
        )

    def set_filter_for_all(self, index):
        """
        Asigna un filtro a todos los rostros.
        """
        self.default_filter_index = (
            self.manager.normalize(index)
        )

        for face_id in list(
            self.filter_indices
        ):
            self.filter_indices[face_id] = (
                self.default_filter_index
            )

    def next_filter_for_all(self):
        """
        Selecciona el siguiente filtro.
        """
        self.set_filter_for_all(
            self.default_filter_index + 1
        )

    def previous_filter_for_all(self):
        """
        Selecciona el filtro anterior.
        """
        self.set_filter_for_all(
            self.default_filter_index - 1
        )

    def recalibrate(self):
        """
        Reinicia los detectores de movimiento vertical.
        """
        for detector in (
            self.motion_detectors.values()
        ):
            detector.reset()

    def set_sensitivity(self, sensitivity):
        """
        Ajusta en caliente qué tan fácil es disparar un cambio de filtro
        con el movimiento vertical. Aplica tanto a los detectores ya
        existentes como a los que se creen para nuevos rostros.
        """
        self.sensitivity = min(2.0, max(0.5, sensitivity))
        for detector in self.motion_detectors.values():
            detector.set_sensitivity(self.sensitivity)

    def _cleanup_state(self):
        """
        Elimina información de rostros que desaparecieron.
        """
        active_ids = self.tracker.active_ids

        mappings = (
            self.motion_detectors,
            self.filter_indices,
        )

        for mapping in mappings:
            expired = [
                face_id
                for face_id in mapping
                if face_id not in active_ids
            ]

            for face_id in expired:
                del mapping[face_id]

    def process(self, frame):
        """
        Procesa un fotograma completo.
        """
        height, width = frame.shape[:2]

        # Optimización de rendimiento: se detecta sobre una copia reducida
        # cuando el fotograma es más ancho que DETECTION_MAX_WIDTH. Los
        # landmarks de MediaPipe son coordenadas normalizadas (0-1), así
        # que siguen siendo válidos al convertirlos usando el ancho/alto
        # ORIGINALES del fotograma (más abajo, en get_key_points).
        if width > DETECTION_MAX_WIDTH:
            scale = DETECTION_MAX_WIDTH / float(width)
            detection_frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_frame = frame

        rgb = cv2.cvtColor(
            detection_frame,
            cv2.COLOR_BGR2RGB,
        )

        rgb.flags.writeable = False

        results = self.face_mesh.process(
            rgb
        )

        rgb.flags.writeable = True

        faces = get_all_face_landmarks(
            results
        )

        detections = []

        for face_landmarks in faces:
            key_points = get_key_points(
                face_landmarks,
                width,
                height,
            )

            bbox = get_face_bbox(
                key_points["all"],
                width,
                height,
                padding=18,
            )

            detections.append(
                {
                    "landmarks": face_landmarks,
                    "key_points": key_points,
                    "bbox": bbox,
                    "area": bbox_area(bbox),
                }
            )

        track_ids = self.tracker.update(
            [
                item["bbox"]
                for item in detections
            ]
        )

        details = []

        for detection, face_id in zip(
            detections,
            track_ids,
        ):
            key_points = (
                detection["key_points"]
            )

            bbox = detection["bbox"]

            x1, y1, x2, y2 = bbox

            face_height = max(
                1,
                y2 - y1,
            )

            detector = (
                self.motion_detectors.setdefault(
                    face_id,
                    VerticalMotionDetector(
                        threshold_ratio=0.11,
                        cooldown_seconds=0.85,
                        smoothing_alpha=0.35,
                        sensitivity=self.sensitivity,
                    ),
                )
            )

            filter_index = (
                self.filter_indices.setdefault(
                    face_id,
                    self.default_filter_index,
                )
            )

            (
                direction,
                delta,
                threshold,
            ) = detector.update(
                key_points["nose"][1],
                face_height,
            )

            if self.auto_change:
                if direction == "ARRIBA":
                    filter_index = (
                        self.manager.normalize(
                            filter_index + 1
                        )
                    )

                elif direction == "ABAJO":
                    filter_index = (
                        self.manager.normalize(
                            filter_index - 1
                        )
                    )

                self.filter_indices[
                    face_id
                ] = filter_index

            (
                filter_info,
                filter_image,
            ) = self.manager.get(
                filter_index
            )

            # --------------------------------------------
            # FILTRO SONRISA DE GUASÓN
            # --------------------------------------------

            if (
                filter_info["type"]
                == "joker_smile"
            ):
                apply_joker_smile(
                    frame,
                    key_points,
                    bbox,
                )

            # --------------------------------------------
            # FILTRO OJOS GRANDES
            # --------------------------------------------

            elif (
                filter_info["type"]
                == "big_eyes"
            ):
                apply_big_eyes(
                    frame,
                    key_points,
                    bbox,
                )

            # --------------------------------------------
            # FILTRO CORAZONES EN LOS OJOS
            # --------------------------------------------

            elif (
                filter_info["type"]
                == "heart_eyes"
            ):
                apply_heart_eyes(
                    frame,
                    key_points,
                )

            # --------------------------------------------
            # FILTROS PNG (con rotación según la inclinación
            # lateral de la cabeza)
            # --------------------------------------------

            else:
                fx, fy, fw, fh = (
                    compute_filter_position(
                        filter_info["type"],
                        filter_image,
                        key_points,
                        bbox,
                    )
                )

                head_roll = roll_angle_degrees(
                    key_points["left_eye"],
                    key_points["right_eye"],
                )

                overlay_png(
                    frame,
                    filter_image,
                    fx,
                    fy,
                    fw,
                    fh,
                    angle_degrees=head_roll,
                )

            if self.show_landmarks:
                self.drawing.draw_landmarks(
                    image=frame,
                    landmark_list=(
                        detection["landmarks"]
                    ),
                    connections=(
                        self.mesh_connections
                    ),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=(
                        self.drawing_styles
                        .get_default_face_mesh_tesselation_style()
                    ),
                )

            if direction == "ESTABLE":
                box_color = (0, 220, 0)
            else:
                box_color = (0, 190, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            cv2.circle(
                frame,
                key_points["nose"],
                4,
                (0, 255, 255),
                -1,
            )

            label_y = (
                y1 - 12
                if y1 > 30
                else y1 + 24
            )

            label_text = (
                f"Rostro {face_id} | "
                f"{filter_info['name']} | "
                f"{direction}"
            )

            put_label(
                frame,
                label_text,
                (x1, label_y),
                scale=0.48,
                thickness=1,
            )

            details.append(
                {
                    "id": face_id,
                    "bbox": bbox,
                    "filter_index": (
                        filter_index
                    ),
                    "filter_name": (
                        filter_info["name"]
                    ),
                    "direction": direction,
                    "delta": delta,
                    "threshold": threshold,
                }
            )

        self._cleanup_state()

        fps = self.fps_counter.update()

        return frame, {
            "face_count": len(details),
            "fps": fps,
            "faces": details,
            "max_num_faces": (
                self.max_num_faces
            ),
        }

    def close(self):
        """
        Libera los recursos de MediaPipe.
        """
        self.face_mesh.close()