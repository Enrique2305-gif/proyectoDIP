# Filtros faciales — Sprint 3 (demo mejorada)

Prototipo con detección de varios rostros, landmarks, movimiento vertical, filtros y una interfaz demostrativa.

> Esta versión incorpora mejoras sobre la entregada en Sprint 3: sensibilidad de
> movimiento configurable, seguimiento multirrostro más estable, dos filtros
> nuevos (Lentes de sol y Corazones), rotación de los filtros PNG según la
> inclinación de la cabeza, una optimización de rendimiento para varios
> rostros y una interfaz visual renovada. Ver `CAMBIOS.md` para el detalle.

## Requisito importante

El código usa `mp.solutions.face_mesh`, por lo que debe ejecutarse con **Python 3.12 de 64 bits** y `mediapipe==0.10.21`.

No cambies MediaPipe a `0.10.30` o superior sin adaptar el código a la API Tasks.

## Instalación rápida en Windows

Abre PowerShell dentro de la carpeta del proyecto y ejecuta:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\instalar_windows.ps1
```

## Instalación manual

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --no-cache-dir mediapipe==0.10.21
python -m pip install --no-cache-dir -r requirements.txt
```

Comprueba la instalación:

```powershell
python -c "import cv2, mediapipe as mp; print(cv2.__version__); print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

El último valor debe ser `True`.

## Ejecución

Landmarks de varios rostros:

```powershell
python fase1_landmarks.py
```

Filtros de varios rostros:

```powershell
python fase3_filtros.py
```

Interfaz demo:

```powershell
python demo_interfaz.py
```

## Controles de la fase 3

- `A`: filtro anterior.
- `D`: filtro siguiente.
- `+` / `-`: subir o bajar la sensibilidad del movimiento.
- `M`: mostrar u ocultar landmarks.
- `R`: recalibrar el movimiento.
- `S`: guardar captura.
- `Q`: salir.

## Controles de la fase 2

- `+` / `-`: subir o bajar la sensibilidad del movimiento.
- `R`: recalibrar la referencia.
- `S`: guardar captura.
- `Q`: salir.
