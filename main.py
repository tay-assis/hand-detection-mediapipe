import mediapipe as mp
import cv2
import csv
import os

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
    
    # Save the landmarks to a CSV file
    # def save_landmarks(self, results):
    #     with open(self.file_path, mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #         if file.tell() == 0:
    #             writer.writerow(['Hand', 'Landmark', 'x', 'y', 'z'])
    #         if results.multi_hand_landmarks:
    #             for hand_landmarks in results.multi_hand_landmarks:
    #                 for idx, landmark in enumerate(hand_landmarks.landmark):
    #                     hand_label = 'Right' if results.multi_handedness[0].classification[0].label == 'Right' else 'Left'
    #                     writer.writerow([hand_label, idx, landmark.x, landmark.y, landmark.z])

    # Draw the points on the image
    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Run the hand detection
    def run(self):
        while self.cap.isOpened():
            image, results = self.process_frame()
            if image is None:
                break
            # self.save_landmarks(results)
            self.draw_landmarks(image, results)
            cv2.imshow('Hand Detection', image)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    hand_detection = HandDetection()
    hand_detection.run()
