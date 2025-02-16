import mediapipe as mp
import cv2
import csv
import os
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcan input
cap = cv2.VideoCapture(0)

# Make a detection hands pose
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = hands.process(image)

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
                writer.writerow(['Hand', 'Landmark', 'x', 'y', 'z'])

            # Write hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        writer.writerow(['Right' if results.multi_handedness[0].classification[0].label == 'Right' else 'Left', idx, landmark.x, landmark.y, landmark.z])

                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
