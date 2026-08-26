import time


class VerticalMotionDetector:
    """
    Detecta movimientos verticales con suavizado exponencial, umbral relativo
    al tamaño del rostro y tiempo de espera entre cambios.

    Incorpora un control de "sensibilidad" (0.5 a 2.0) que permite ajustar,
    en tiempo real, qué tan fácil es disparar un cambio de filtro:

    - Sensibilidad baja (0.5): se necesita un movimiento más amplio y hay
      más tiempo de espera entre cambios. Útil si el sistema detecta
      cambios "falsos" con solo mover un poco la cabeza.
    - Sensibilidad alta (2.0): reacciona con un movimiento pequeño y un
      tiempo de espera corto. Útil para una demo más ágil.
    """

    MIN_SENSITIVITY = 0.5
    MAX_SENSITIVITY = 2.0

    def __init__(
        self,
        threshold_ratio=0.11,
        cooldown_seconds=0.85,
        smoothing_alpha=0.35,
        minimum_threshold=16,
        sensitivity=1.0,
    ):
        # Valores base (sensibilidad = 1.0), tal como en la versión original.
        self.base_threshold_ratio = threshold_ratio
        self.base_cooldown_seconds = cooldown_seconds
        self.base_minimum_threshold = minimum_threshold
        self.smoothing_alpha = smoothing_alpha

        self.sensitivity = sensitivity
        self.threshold_ratio = threshold_ratio
        self.cooldown_seconds = cooldown_seconds
        self.minimum_threshold = minimum_threshold
        self._apply_sensitivity()

        self.base_y = None
        self.smoothed_y = None
        self.last_change_time = 0.0
        self.last_direction = "ESTABLE"

    def _apply_sensitivity(self):
        """Recalcula umbral y cooldown en función de la sensibilidad actual."""
        # A mayor sensibilidad, menor umbral y menor tiempo de espera.
        factor = 1.0 / max(0.3, self.sensitivity)
        self.threshold_ratio = self.base_threshold_ratio * factor
        self.minimum_threshold = max(
            6, self.base_minimum_threshold * factor
        )
        self.cooldown_seconds = max(
            0.20, self.base_cooldown_seconds * factor
        )

    def set_sensitivity(self, sensitivity):
        """Permite ajustar la sensibilidad en caliente (p. ej. desde un slider)."""
        self.sensitivity = min(
            self.MAX_SENSITIVITY,
            max(self.MIN_SENSITIVITY, sensitivity),
        )
        self._apply_sensitivity()

    def reset(self, current_y=None):
        self.base_y = float(current_y) if current_y is not None else None
        self.smoothed_y = float(current_y) if current_y is not None else None
        self.last_direction = "ESTABLE"

    def update(self, current_y, face_height):
        current_y = float(current_y)

        if self.smoothed_y is None:
            self.smoothed_y = current_y
        else:
            self.smoothed_y = (
                self.smoothing_alpha * current_y
                + (1.0 - self.smoothing_alpha) * self.smoothed_y
            )

        threshold = max(
            self.minimum_threshold,
            int(face_height * self.threshold_ratio),
        )

        if self.base_y is None:
            self.base_y = self.smoothed_y
            return "ESTABLE", 0.0, threshold

        delta = self.smoothed_y - self.base_y
        now = time.time()
        direction = "ESTABLE"

        if now - self.last_change_time >= self.cooldown_seconds:
            # En una imagen, el eje Y aumenta hacia abajo.
            if delta < -threshold:
                direction = "ARRIBA"
            elif delta > threshold:
                direction = "ABAJO"

            if direction != "ESTABLE":
                self.last_change_time = now
                self.base_y = self.smoothed_y

        # Cuando la cabeza regresa cerca del centro, la referencia se adapta
        # lentamente para compensar movimientos naturales y pequeños errores.
        if abs(delta) < threshold * 0.35:
            self.base_y = 0.98 * self.base_y + 0.02 * self.smoothed_y

        self.last_direction = direction
        return direction, delta, threshold
