import mediapipe as mp
import cv2
import os
from calculation_amplitude import CalculationAmplitudeClass
from vector_drawer import VectorDrawer

class HandDetection:
    # Inicializa a classe HandDetection
    def __init__(self, finger1='THUMB_TIP', finger2='INDEX_FINGER_TIP', min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.finger1 = finger1
        self.finger2 = finger2
        
        # Detecta apenas uma mão.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=min_detection_confidence,
                                                 min_tracking_confidence=min_tracking_confidence)
        self.cap = cv2.VideoCapture(0)
        # self.data_dir = 'hand_positions'
        # os.makedirs(self.data_dir, exist_ok=True)
        # self.file_path = os.path.join(self.data_dir, 'hand_landmarks.csv')
        self.calc_amplitude = CalculationAmplitudeClass()
        self.vector_drawer = VectorDrawer()
    
    # Processa cada frame do programa enquanto a câmera estiver ativa
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
    
    # Desenha os pontos das mãos detectados 
    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                finger1_landmark = hand_landmarks.landmark[getattr(mp.solutions.hands.HandLandmark, self.finger1)]
                finger2_landmark = hand_landmarks.landmark[getattr(mp.solutions.hands.HandLandmark, self.finger2)]
                
                # Desenha os vetores formados pelos membros
                self.vector_drawer.draw_vector(image, wrist, finger1_landmark, color=(0, 255, 0))
                self.vector_drawer.draw_vector(image, wrist, finger2_landmark, color=(0, 255, 0))
                
                # Calcula o ângulo e a amplitude de movimento
                vector_1 = self.calc_amplitude.create_vector(wrist, finger1_landmark)
                vector_2 = self.calc_amplitude.create_vector(wrist, finger2_landmark)
                angle = self.calc_amplitude.calculate_amplitude(vector_1, vector_2)
                
                # O resultado do ângulo é exibido
                cv2.putText(image, f'Ângulo: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Executa a detecção da mão através da câmera
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