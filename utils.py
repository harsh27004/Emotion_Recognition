import cv2
import mediapipe as mp
import numpy as np

# Initialize FaceMesh once to improve performance
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5
)

def get_face_landmarks(image, draw=False):
    print("Function get_face_landmarks() called")

    if image is None:
        print("Warning: Image is None")
        return None

    # Resize and Convert image to RGB (needed for Mediapipe)
    image_resized = cv2.resize(image, (640, 480))
    image_input_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_input_rgb = np.ascontiguousarray(image_input_rgb)

    # Process the image with FaceMesh
    results = face_mesh.process(image_input_rgb)
    image_landmarks = []

    if results.multi_face_landmarks:
        ls_single_face = results.multi_face_landmarks[0].landmark

        xs_, ys_, zs_ = [], [], []
        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)

        # Avoid `min()` errors by checking if lists are non-empty
        if xs_ and ys_ and zs_:
            for j in range(len(xs_)):
                image_landmarks.append(xs_[j] - min(xs_))
                image_landmarks.append(ys_[j] - min(ys_))
                image_landmarks.append(zs_[j] - min(zs_))
        else:
            print("Warning: No face landmarks detected")
            return None

        # Draw landmarks if needed
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image_resized,  # Use resized image for drawing
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        return image_landmarks  # Return extracted landmarks

    print("Warning: No face detected")
    return None  # Return None if no face is detected
