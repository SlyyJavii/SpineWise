import os
import cv2 as cv
from datetime import datetime

# Lock base folder to the true posture_images folder inside /data
POSTURE_IMAGE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "data", "posture_images"
))

def initialize_folders():
    for label in ["good", "bad", "moderate"]:
        folder = os.path.join(POSTURE_IMAGE_ROOT, label)
        os.makedirs(folder, exist_ok=True)
    print(f"[INIT] ✅ Using folder: {POSTURE_IMAGE_ROOT}")

def save_posture_image(frame, label):
    folder_path = os.path.join(POSTURE_IMAGE_ROOT, label)

    if not os.path.exists(folder_path):
        print(f"[ERROR] ❌ Folder does not exist: {folder_path}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    filename = os.path.join(folder_path, f"{label}_{timestamp}.jpg")
    success = cv.imwrite(filename, frame)

    if success:
        print(f"[LOG] ✅ Image saved to {filename}")
    else:
        print(f"[ERROR] ❌ Failed to save image to {filename}")

