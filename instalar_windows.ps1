$ErrorActionPreference = "Stop"

Write-Host "Verificando Python 3.12..." -ForegroundColor Cyan
try {
    py -3.12 --version
} catch {
    Write-Host "No se encontro Python 3.12." -ForegroundColor Red
    Write-Host "Instalalo y vuelve a ejecutar este archivo." -ForegroundColor Yellow
    exit 1
}

if (Test-Path ".venv") {
    Write-Host "Eliminando entorno virtual anterior..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

Write-Host "Creando entorno virtual con Python 3.12..." -ForegroundColor Cyan
py -3.12 -m venv .venv

Write-Host "Actualizando pip..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "Instalando dependencias..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install --no-cache-dir -r requirements.txt

Write-Host "Comprobando OpenCV y MediaPipe..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -c "import cv2, mediapipe as mp; print('OpenCV:', cv2.__version__); print('MediaPipe:', mp.__version__); print('API solutions:', hasattr(mp, 'solutions'))"

Write-Host "Instalacion finalizada." -ForegroundColor Green
Write-Host "Ejecuta: .\.venv\Scripts\python.exe fase1_landmarks.py"
