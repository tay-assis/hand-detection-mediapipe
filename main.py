import mediapipe as mp
import cv2
# import csv
import os
import math

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
        self.calc_amplitude = self.CalculationAmplitudeClass()
        self.vector_drawer = self.VectorDrawer()
    
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
    
    class CalculationAmplitudeClass:
        # This class is responsible for calculating the amplitude of the hand movement
        def create_vector(self, coord1, coord2):
            # Make the calculation of the vector
            return (coord2.x - coord1.x, coord2.y - coord1.y)

        def modulation_vector(self, vector):
            # Calculate the modulus of the vector
            return (vector[0]**2 + vector[1]**2) ** 0.5
        
        def convert_degrees(self, radians):
            # Convert radians to degrees
            return radians * 180 / 3.14159265
        
        def calculate_amplitude(self, vector_1, vector_2):
            # Product scalar
            scalar = ((vector_1[0] * vector_2[0]) + (vector_1[1] * vector_2[1]))    

            # Modulus of the vectors
            result_modulus = self.modulation_vector(vector_1) * self.modulation_vector(vector_2)

            # Cosine of the angle between the vectors
            cosine = scalar / result_modulus

            # Return angle in degrees
            return self.convert_degrees(math.acos(cosine))
        
    class VectorDrawer:
        def draw_vector(self, image, coord1, coord2, color=(0, 255, 0), thickness=2):
            # Draw the conection points on the image
            start_point = (int(coord1.x * image.shape[1]), int(coord1.y * image.shape[0]))
            end_point = (int(coord2.x * image.shape[1]), int(coord2.y * image.shape[0]))
            cv2.line(image, start_point, end_point, color, thickness)
        
if __name__ == "__main__":
    hand_detection = HandDetection()
    hand_detection.run()
