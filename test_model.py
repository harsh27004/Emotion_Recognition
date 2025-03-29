import pickle
import cv2
from utils import get_face_landmarks

# Emotion Labels
emotions = ['HAPPY', 'SAD', 'SURPRISED']

# Load Model
with open('./model', 'rb') as f:
    model = pickle.load(f)

# Open Camera
cap = cv2.VideoCapture(0)  # Try 0 if 2 doesn't work

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame!")
        break

    # Get Face Landmarks
    face_landmarks = get_face_landmarks(frame, draw=True)

    if not face_landmarks:
        print("No face detected, skipping frame.")
        continue
    
    print(f"🔹 Extracted {len(face_landmarks)} landmarks: {face_landmarks[:10]}...")  # Print first 10 for reference


    # Predict Emotion
    output = model.predict([face_landmarks])
    print(f"Predicted: {output}, Mapped to: {emotions[int(output[0])]}")
    # Display Text on Frame
    cv2.putText(frame,
                emotions[int(output[0])],
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                5)

    # Show Frame
    cv2.imshow('Emotion Detection', frame)

    # Press 'ESC' to Exit
    if cv2.waitKey(25) & 0xFF == 27:
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
