import mediapipe as mp
import cv2
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Webcan input
cap = cv2.VideoCapture(0)

# Make a detection hands pose
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Define the directory and file path
        data_dir = 'hand_positions'
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, 'hand_landmarks.csv')

        # Open CSV file in append mode
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header if file is empty
            if file.tell() == 0:
                writer.writerow(['Landmark', 'x', 'y', 'z'])

            # Write right hand landmarks
            if results.right_hand_landmarks:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    writer.writerow([f'Landmark {idx}', landmark.x, landmark.y, landmark.z])

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                 )

        cv2.imshow('Hand Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()