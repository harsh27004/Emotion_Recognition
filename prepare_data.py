import os
import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = './data'

output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        # Load image and check if it's valid
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load {image_path}. Skipping...")
            continue

        # Resize image to improve face detection
        image = cv2.resize(image, (256, 256))

        # Get landmarks
        face_landmarks = get_face_landmarks(image)
        if face_landmarks is None:
            print(f"Skipping {image_path}: No landmarks detected")
            continue

        # Ensure landmarks meet the expected format
        if len(face_landmarks) > 1000:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

# Save only if data exists
if output:
    np.savetxt('data.txt', np.asarray(output))
    print(" Data successfully saved.")
else:
    print(" No valid data to save.")
