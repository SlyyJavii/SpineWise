import os 
import cv2 as cv
from datetime import datetime
# Creation of required directories ( executed once when imported)
def initialize_folders(base_path="posture_images"):
    # Defines a function that creates subdirectories under a base folder.
    # The base folder is "posture_images" by default.
    for label in ["good","bad","moderate"]: # Iterates through a list of labels.
        os.makedirs(os.path.join(base_path,label),exist_ok=True) # Creates a directory for each label if it does not already exist.
    print(f"[INIT] Posture image folders ready at {base_path}/") # Prints a message indicating that the folders are ready.
# Save image to correct label folder 
def save_posture_image(frame, label, base_path="posture_images"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")  # Unique to the microsecond
    filename = os.path.join(base_path, label, f"{label}_{timestamp}.jpg")
    cv.imwrite(filename, frame)
    print(f"[LOG] Image saved to {filename}")
