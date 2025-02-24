import mediapipe as mp
import cv2
import os
from calculation_amplitude import CalculationAmplitudeClass
from vector_drawer import VectorDrawer

class HandDetection:
    # Initialize the HandDetection class
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=min_detection_confidence,
                                                 min_tracking_confidence=min_tracking_confidence)
        self.cap = cv2.VideoCapture(0)
        self.data_dir = 'hand_positions'
        os.makedirs(self.data_dir, exist_ok=True)
        self.file_path = os.path.join(self.data_dir, 'hand_landmarks.csv')
        self.calc_amplitude = CalculationAmplitudeClass()
        self.vector_drawer = VectorDrawer()
    
    # Process the frame
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mp_hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    # Draw the points on the image
    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                index_finger = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                thumb = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                
                # Draw vectors
                self.vector_drawer.draw_vector(image, wrist, index_finger, color=(0, 255, 0))
                self.vector_drawer.draw_vector(image, wrist, thumb, color=(0, 255, 0))
                
                # Calculate amplitude
                vector_1 = self.calc_amplitude.create_vector(wrist, index_finger)
                vector_2 = self.calc_amplitude.create_vector(wrist, thumb)
                angle = self.calc_amplitude.calculate_amplitude(vector_1, vector_2)
                
                # Display angle
                cv2.putText(image, f'Angle: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Run the hand detection
    def run(self):
        while self.cap.isOpened():
            image, results = self.process_frame()
            if image is None:
                break
            self.draw_landmarks(image, results)
            cv2.imshow('Hand Detection', image)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()