from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


def save_png(name, image):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / name
    cv2.imwrite(str(path), image)
    print(f"Creado: {path}")


def create_glasses():
    image = np.zeros((220, 600, 4), dtype=np.uint8)
    cv2.ellipse(image, (200, 110), (120, 75), 0, 0, 360, (20, 20, 20, 255), 12)
    cv2.ellipse(image, (400, 110), (120, 75), 0, 0, 360, (20, 20, 20, 255), 12)
    cv2.line(image, (320, 110), (280, 110), (20, 20, 20, 255), 10)
    cv2.line(image, (80, 100), (20, 70), (20, 20, 20, 255), 8)
    cv2.line(image, (520, 100), (580, 70), (20, 20, 20, 255), 8)
    cv2.ellipse(image, (200, 110), (105, 60), 0, 0, 360, (180, 220, 255, 70), -1)
    cv2.ellipse(image, (400, 110), (105, 60), 0, 0, 360, (180, 220, 255, 70), -1)
    return image


def create_mustache():
    image = np.zeros((220, 520, 4), dtype=np.uint8)
    cv2.ellipse(image, (230, 120), (145, 60), -10, 0, 360, (25, 25, 25, 255), -1)
    cv2.ellipse(image, (290, 120), (145, 60), 10, 0, 360, (25, 25, 25, 255), -1)
    cv2.circle(image, (260, 110), 35, (25, 25, 25, 255), -1)
    cv2.ellipse(image, (110, 110), (120, 55), -25, 180, 360, (25, 25, 25, 255), -1)
    cv2.ellipse(image, (410, 110), (120, 55), 25, 180, 360, (25, 25, 25, 255), -1)
    return image


def create_crown():
    image = np.zeros((300, 600, 4), dtype=np.uint8)
    points = np.array(
        [[70, 250], [130, 80], [230, 210], [300, 50], [370, 210], [470, 80], [530, 250]],
        np.int32,
    )
    cv2.fillPoly(image, [points], (0, 190, 255, 255))
    cv2.polylines(image, [points], True, (0, 120, 210, 255), 8)
    for point in [(130, 80), (300, 50), (470, 80)]:
        cv2.circle(image, point, 28, (0, 255, 255, 255), -1)
        cv2.circle(image, point, 28, (0, 120, 210, 255), 4)
    cv2.rectangle(image, (70, 235), (530, 285), (0, 170, 255, 255), -1)
    cv2.rectangle(image, (70, 235), (530, 285), (0, 120, 210, 255), 6)
    return image


def create_sunglasses():
    image = np.zeros((220, 600, 4), dtype=np.uint8)
    # Lentes oscuros sólidos (a diferencia de "Gafas", que son transparentes).
    cv2.ellipse(image, (200, 110), (120, 75), 0, 0, 360, (12, 12, 12, 255), -1)
    cv2.ellipse(image, (400, 110), (120, 75), 0, 0, 360, (12, 12, 12, 255), -1)
    cv2.ellipse(image, (200, 110), (120, 75), 0, 0, 360, (0, 0, 0, 255), 8)
    cv2.ellipse(image, (400, 110), (120, 75), 0, 0, 360, (0, 0, 0, 255), 8)
    cv2.line(image, (320, 110), (280, 110), (0, 0, 0, 255), 10)
    cv2.line(image, (80, 100), (20, 70), (0, 0, 0, 255), 8)
    cv2.line(image, (520, 100), (580, 70), (0, 0, 0, 255), 8)
    # Brillo tipo espejo, para que no se vean lentes totalmente planos.
    cv2.ellipse(image, (165, 88), (34, 16), -20, 0, 360, (130, 170, 255, 150), -1)
    cv2.ellipse(image, (365, 88), (34, 16), -20, 0, 360, (130, 170, 255, 150), -1)
    return image


def create_cat_ears():
    image = np.zeros((330, 700, 4), dtype=np.uint8)
    left = np.array([[140, 300], [210, 50], [330, 300]], np.int32)
    right = np.array([[370, 300], [490, 50], [560, 300]], np.int32)
    cv2.fillPoly(image, [left], (35, 35, 35, 255))
    cv2.fillPoly(image, [right], (35, 35, 35, 255))
    cv2.polylines(image, [left], True, (0, 0, 0, 255), 8)
    cv2.polylines(image, [right], True, (0, 0, 0, 255), 8)
    left_inner = np.array([[185, 260], [215, 115], [285, 260]], np.int32)
    right_inner = np.array([[415, 260], [485, 115], [515, 260]], np.int32)
    cv2.fillPoly(image, [left_inner], (180, 120, 180, 255))
    cv2.fillPoly(image, [right_inner], (180, 120, 180, 255))
    return image


if __name__ == "__main__":
    save_png("glasses.png", create_glasses())
    save_png("mustache.png", create_mustache())
    save_png("crown.png", create_crown())
    save_png("cat_ears.png", create_cat_ears())
    save_png("sunglasses.png", create_sunglasses())
