from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from face_utils import put_label, save_frame
from filter_engine import FaceFilterEngine


# ============================================================
# PALETA DE COLORES (tokens de diseño reutilizados en toda la interfaz)
# ============================================================

COLOR_BG = "#0b1220"          # fondo general / video
COLOR_HEADER = "#0a0f1e"      # encabezado
COLOR_PANEL = "#151d30"       # panel lateral
COLOR_CARD = "#1c263d"        # tarjetas dentro del panel
COLOR_CARD_BORDER = "#2b3856"
COLOR_ACCENT = "#6366f1"      # acento principal (índigo)
COLOR_ACCENT_DARK = "#4338ca"
COLOR_TEXT = "#f8fafc"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_SUCCESS = "#22c55e"
COLOR_DANGER = "#ef4444"


def sensitivity_label(value):
    """Traduce el valor numérico de sensibilidad a una etiqueta legible."""
    if value < 0.85:
        return "Baja"
    if value < 1.35:
        return "Media"
    return "Alta"


class FaceFiltersDemoApp:
    """Interfaz demostrativa del Sprint 3; no representa el producto final."""

    def __init__(self, root):
        self.root = root
        self.root.title("Filtros faciales - Demo Sprint 3")
        self.root.geometry("1200x740")
        self.root.minsize(1000, 660)
        self.root.configure(bg=COLOR_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.capture = None
        self.running = False
        self.current_frame = None
        self.engine = FaceFilterEngine(max_num_faces=5)

        self.face_count_var = tk.StringVar(value="Rostros detectados: 0")
        self.fps_var = tk.StringVar(value="FPS: 0.0")
        self.status_var = tk.StringVar(value="Cámara detenida")
        self.sensitivity_display_var = tk.StringVar(value="Media (1.00x)")
        self.landmarks_var = tk.BooleanVar(value=False)
        self.auto_change_var = tk.BooleanVar(value=True)
        self.sensitivity_var = tk.DoubleVar(value=1.0)

        self.panel_canvas = None
        self.filter_canvas = None
        self.status_dot = None

        self._build_styles()
        self._build_layout()

        # Un solo manejador decide si la rueda desplaza la lista de filtros
        # o el panel completo, según la posición actual del cursor.
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

    # --------------------------------------------------------
    # ESTILOS
    # --------------------------------------------------------

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 9),
            background=COLOR_ACCENT,
            foreground="#ffffff",
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLOR_ACCENT_DARK), ("pressed", COLOR_ACCENT_DARK)],
        )

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            padding=(10, 8),
            background=COLOR_CARD,
            foreground=COLOR_TEXT,
            borderwidth=1,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#28344f"), ("pressed", "#28344f")],
        )

        style.configure(
            "Filter.TButton",
            font=("Segoe UI", 9),
            padding=(8, 6),
            background=COLOR_CARD,
            foreground=COLOR_TEXT,
            borderwidth=1,
        )
        style.map(
            "Filter.TButton",
            background=[("active", COLOR_ACCENT), ("pressed", COLOR_ACCENT_DARK)],
        )

        style.configure(
            "Demo.TCheckbutton",
            background=COLOR_PANEL,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10),
        )
        style.map(
            "Demo.TCheckbutton",
            background=[("active", COLOR_PANEL)],
        )

        style.configure(
            "Demo.Horizontal.TScale",
            background=COLOR_PANEL,
            troughcolor=COLOR_CARD,
        )

        style.configure("Demo.TSeparator", background=COLOR_CARD_BORDER)

    # --------------------------------------------------------
    # LAYOUT GENERAL
    # --------------------------------------------------------

    def _build_layout(self):
        header = tk.Frame(self.root, bg=COLOR_HEADER, height=76)
        header.pack(fill="x")
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg=COLOR_HEADER)
        title_box.pack(anchor="w", padx=24, pady=(12, 0))

        tk.Label(
            title_box,
            text="Aplicación de filtros faciales",
            bg=COLOR_HEADER,
            fg=COLOR_TEXT,
            font=("Segoe UI", 19, "bold"),
        ).pack(side="left")

        tk.Label(
            header,
            text="Prototipo demostrativo · detección de hasta 5 rostros en tiempo real",
            bg=COLOR_HEADER,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=25)

        # Línea de acento bajo el encabezado.
        tk.Frame(self.root, bg=COLOR_ACCENT, height=3).pack(fill="x")

        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=18, pady=18)

        video_container = tk.Frame(body, bg="#020617", bd=1, relief="solid")
        video_container.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(
            video_container,
            text="Presiona “Iniciar cámara” para comenzar",
            bg="#020617",
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 15),
        )
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_scrollable_control_panel(body)

    # --------------------------------------------------------
    # PANEL DE CONTROL (con scroll)
    # --------------------------------------------------------

    def _section_title(self, parent, text):
        """Encabezado de sección con una pequeña barra de acento."""
        row = tk.Frame(parent, bg=COLOR_PANEL)
        row.pack(fill="x", padx=18, pady=(16, 6))
        tk.Frame(row, bg=COLOR_ACCENT, width=4, height=16).pack(side="left", padx=(0, 8))
        tk.Label(
            row,
            text=text,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

    def _build_scrollable_control_panel(self, parent):
        """Construye un panel derecho que también puede desplazarse."""
        panel_shell = tk.Frame(parent, bg=COLOR_PANEL, width=320)
        panel_shell.pack(side="right", fill="y", padx=(16, 0))
        panel_shell.pack_propagate(False)

        self.panel_canvas = tk.Canvas(
            panel_shell,
            bg=COLOR_PANEL,
            highlightthickness=0,
            bd=0,
        )
        panel_scrollbar = ttk.Scrollbar(
            panel_shell,
            orient="vertical",
            command=self.panel_canvas.yview,
        )
        self.panel_canvas.configure(yscrollcommand=panel_scrollbar.set)

        panel_scrollbar.pack(side="right", fill="y")
        self.panel_canvas.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(self.panel_canvas, bg=COLOR_PANEL)
        panel_window = self.panel_canvas.create_window(
            (0, 0),
            window=panel,
            anchor="nw",
        )

        panel.bind(
            "<Configure>",
            lambda _event: self.panel_canvas.configure(
                scrollregion=self.panel_canvas.bbox("all")
            ),
        )
        self.panel_canvas.bind(
            "<Configure>",
            lambda event: self.panel_canvas.itemconfigure(
                panel_window,
                width=event.width,
            ),
        )

        tk.Label(
            panel,
            text="Panel de control",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=18, pady=(18, 10))

        self._build_status_card(panel)

        ttk.Button(
            panel,
            text="Iniciar cámara",
            style="Primary.TButton",
            command=self.start_camera,
        ).pack(fill="x", padx=18, pady=(14, 4))

        ttk.Button(
            panel,
            text="Detener cámara",
            style="Secondary.TButton",
            command=self.stop_camera,
        ).pack(fill="x", padx=18, pady=4)

        ttk.Button(
            panel,
            text="Guardar captura",
            style="Secondary.TButton",
            command=self.take_screenshot,
        ).pack(fill="x", padx=18, pady=4)

        ttk.Button(
            panel,
            text="Recalibrar movimiento",
            style="Secondary.TButton",
            command=self.engine.recalibrate,
        ).pack(fill="x", padx=18, pady=4)

        ttk.Separator(panel, orient="horizontal", style="Demo.TSeparator").pack(
            fill="x",
            padx=18,
            pady=(16, 0),
        )

        self._section_title(panel, "Seleccionar filtro para todos")
        self._build_scrollable_filter_list(panel)

        ttk.Separator(panel, orient="horizontal", style="Demo.TSeparator").pack(
            fill="x",
            padx=18,
            pady=(4, 0),
        )

        self._section_title(panel, "Sensibilidad de movimiento")
        self._build_sensitivity_control(panel)

        ttk.Separator(panel, orient="horizontal", style="Demo.TSeparator").pack(
            fill="x",
            padx=18,
            pady=(4, 0),
        )

        self._section_title(panel, "Opciones")

        ttk.Checkbutton(
            panel,
            text="Mostrar landmarks",
            variable=self.landmarks_var,
            command=self.toggle_landmarks,
            style="Demo.TCheckbutton",
        ).pack(anchor="w", padx=18, pady=(0, 4))

        ttk.Checkbutton(
            panel,
            text="Cambio con movimiento",
            variable=self.auto_change_var,
            command=self.toggle_auto_change,
            style="Demo.TCheckbutton",
        ).pack(anchor="w", padx=18, pady=4)

        tk.Label(
            panel,
            text=(
                "Cada rostro mantiene un filtro propio.\n"
                "Mover la cabeza arriba o abajo cambia\n"
                "el filtro de esa persona. La cabeza\n"
                "inclinada de lado también gira los\n"
                "filtros con imagen (gafas, corona, etc.)."
            ),
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED,
            justify="left",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=18, pady=(14, 20))

    def _build_status_card(self, parent):
        """Tarjeta con el estado de la cámara y los indicadores en vivo."""
        card = tk.Frame(parent, bg=COLOR_CARD, highlightbackground=COLOR_CARD_BORDER,
                         highlightthickness=1)
        card.pack(fill="x", padx=18)

        status_row = tk.Frame(card, bg=COLOR_CARD)
        status_row.pack(fill="x", padx=12, pady=(10, 4))

        self.status_dot = tk.Canvas(
            status_row, width=12, height=12, bg=COLOR_CARD, highlightthickness=0
        )
        self.status_dot.pack(side="left", padx=(0, 8))
        self._draw_status_dot(active=False)

        tk.Label(
            status_row,
            textvariable=self.status_var,
            bg=COLOR_CARD,
            fg=COLOR_TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        for variable in (self.face_count_var, self.fps_var):
            tk.Label(
                card,
                textvariable=variable,
                bg=COLOR_CARD,
                fg=COLOR_TEXT_MUTED,
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 4))

        tk.Frame(card, bg=COLOR_CARD, height=6).pack()

    def _draw_status_dot(self, active):
        if self.status_dot is None:
            return
        self.status_dot.delete("all")
        color = COLOR_SUCCESS if active else COLOR_DANGER
        self.status_dot.create_oval(1, 1, 11, 11, fill=color, outline="")

    def _build_scrollable_filter_list(self, parent):
        """Crea una lista de filtros con barra y rueda de desplazamiento."""
        filter_shell = tk.Frame(parent, bg=COLOR_PANEL, height=190)
        filter_shell.pack(fill="x", padx=18)
        filter_shell.pack_propagate(False)

        self.filter_canvas = tk.Canvas(
            filter_shell,
            bg=COLOR_PANEL,
            highlightthickness=1,
            highlightbackground=COLOR_CARD_BORDER,
            bd=0,
        )
        filter_scrollbar = ttk.Scrollbar(
            filter_shell,
            orient="vertical",
            command=self.filter_canvas.yview,
        )
        self.filter_canvas.configure(yscrollcommand=filter_scrollbar.set)

        filter_scrollbar.pack(side="right", fill="y")
        self.filter_canvas.pack(side="left", fill="both", expand=True)

        filter_buttons = tk.Frame(self.filter_canvas, bg=COLOR_PANEL)
        filter_window = self.filter_canvas.create_window(
            (0, 0),
            window=filter_buttons,
            anchor="nw",
        )

        filter_buttons.bind(
            "<Configure>",
            lambda _event: self.filter_canvas.configure(
                scrollregion=self.filter_canvas.bbox("all")
            ),
        )
        self.filter_canvas.bind(
            "<Configure>",
            lambda event: self.filter_canvas.itemconfigure(
                filter_window,
                width=event.width,
            ),
        )

        # La interfaz obtiene los nombres directamente del FilterManager.
        # Al agregar filtros en filter_engine.py, aparecerán aquí automáticamente.
        for index, filter_name in enumerate(self.engine.manager.names):
            ttk.Button(
                filter_buttons,
                text=filter_name,
                style="Filter.TButton",
                command=lambda selected=index: self.engine.set_filter_for_all(selected),
            ).pack(fill="x", padx=(0, 4), pady=3)

        tk.Label(
            parent,
            text="Usa la rueda del mouse o la barra lateral.",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=18, pady=(4, 0))

    def _build_sensitivity_control(self, parent):
        """Slider para ajustar en caliente qué tan fácil dispara un cambio de filtro."""
        row = tk.Frame(parent, bg=COLOR_PANEL)
        row.pack(fill="x", padx=18, pady=(0, 4))

        tk.Label(
            row,
            textvariable=self.sensitivity_display_var,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        scale = ttk.Scale(
            parent,
            from_=0.5,
            to=2.0,
            orient="horizontal",
            variable=self.sensitivity_var,
            style="Demo.Horizontal.TScale",
            command=self._on_sensitivity_change,
        )
        scale.pack(fill="x", padx=18, pady=(0, 2))

        labels_row = tk.Frame(parent, bg=COLOR_PANEL)
        labels_row.pack(fill="x", padx=18, pady=(0, 4))
        tk.Label(
            labels_row, text="Menos sensible", bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8),
        ).pack(side="left")
        tk.Label(
            labels_row, text="Más sensible", bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8),
        ).pack(side="right")

        tk.Label(
            parent,
            text=(
                "Baja: hace falta un movimiento más amplio.\n"
                "Alta: reacciona con un movimiento pequeño."
            ),
            bg=COLOR_PANEL,
            fg=COLOR_TEXT_MUTED,
            justify="left",
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=18, pady=(2, 0))

    # --------------------------------------------------------
    # UTILIDADES DE SCROLL
    # --------------------------------------------------------

    @staticmethod
    def _is_descendant(widget, ancestor):
        """Indica si widget pertenece visualmente al contenedor ancestor."""
        current = widget
        while current is not None:
            if current == ancestor:
                return True
            parent_name = current.winfo_parent()
            if not parent_name:
                break
            try:
                current = current.nametowidget(parent_name)
            except KeyError:
                break
        return False

    def _on_mousewheel(self, event):
        """Desplaza la lista de filtros o el panel según dónde esté el cursor."""
        pointer_x, pointer_y = self.root.winfo_pointerxy()
        widget = self.root.winfo_containing(pointer_x, pointer_y)
        if widget is None:
            return

        units = -1 if event.delta > 0 else 1

        if self.filter_canvas and self._is_descendant(widget, self.filter_canvas):
            self.filter_canvas.yview_scroll(units, "units")
            return

        if self.panel_canvas and self._is_descendant(widget, self.panel_canvas):
            self.panel_canvas.yview_scroll(units, "units")

    # --------------------------------------------------------
    # CÁMARA Y PROCESAMIENTO
    # --------------------------------------------------------

    def start_camera(self):
        if self.running:
            return

        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror(
                "Error de cámara",
                "No se pudo abrir la cámara. Revisa los permisos o cambia el índice a 1.",
            )
            return

        self.running = True
        self.status_var.set("Cámara activa")
        self._draw_status_dot(active=True)
        self._update_video()

    def stop_camera(self):
        self.running = False
        self.status_var.set("Cámara detenida")
        self.face_count_var.set("Rostros detectados: 0")
        self.fps_var.set("FPS: 0.0")
        self._draw_status_dot(active=False)

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def toggle_landmarks(self):
        self.engine.show_landmarks = self.landmarks_var.get()

    def toggle_auto_change(self):
        self.engine.auto_change = self.auto_change_var.get()

    def _on_sensitivity_change(self, _value):
        value = self.sensitivity_var.get()
        self.engine.set_sensitivity(value)
        self.sensitivity_display_var.set(
            f"{sensitivity_label(value)} ({value:.2f}x)"
        )

    def take_screenshot(self):
        if self.current_frame is None:
            messagebox.showinfo("Captura", "Primero inicia la cámara.")
            return

        path = save_frame(self.current_frame, "demo_sprint3")
        self.status_var.set(f"Captura guardada: {Path(path).name}")

    def _update_video(self):
        if not self.running or self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Error", "Se perdió la señal de la cámara.")
            return

        frame = cv2.flip(frame, 1)
        processed, info = self.engine.process(frame)
        self.current_frame = processed.copy()

        put_label(
            processed,
            "DEMO SPRINT 3 - PROTOTIPO MULTIRROSTRO",
            (18, 30),
            0.62,
        )

        self.face_count_var.set(f"Rostros detectados: {info['face_count']}")
        self.fps_var.set(f"FPS: {info['fps']:.1f}")

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        available_width = max(self.video_label.winfo_width() - 16, 640)
        available_height = max(self.video_label.winfo_height() - 16, 480)
        image.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

        self.root.after(15, self._update_video)

    def close(self):
        self.stop_camera()
        self.engine.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    FaceFiltersDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
