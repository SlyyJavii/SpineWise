import cv2 as cv
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

cap = cv.VideoCapture(0)
cv.namedWindow('Face Mesh Detection')

# FaceMesh setup
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris landmarks too
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        cv.imshow('Face Mesh Detection', frame)
        if cv.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()
