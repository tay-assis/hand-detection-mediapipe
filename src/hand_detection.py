import ctypes
from email.mime import image
import sys
from unittest import result
import mediapipe as mp
import cv2
import os
import numpy as np
from calculation_amplitude import CalculationAmplitudeClass
from vector_drawer import VectorDrawer

def set_window_icon(window_name, icon_path):
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        hicon = ctypes.windll.user32.LoadImageW(None, icon_path, 1, 0, 0, 0x00000010)
        ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, hicon)  # WM_SETICON (small)
        ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, hicon)  # WM_SETICON (large)

class HandDetection:
    # Inicializa a classe HandDetection
    def __init__(self, finger1='THUMB_TIP', finger2='INDEX_FINGER_TIP', min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.finger1 = finger1
        self.finger2 = finger2

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Detecta apenas uma mão.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=min_detection_confidence,
                                                 min_tracking_confidence=min_tracking_confidence)
        self.cap = cv2.VideoCapture(0)
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
    
    def background_color(self, image, results=None):
        # Converte para RGB, que é o formato que o MediaPipe espera
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Segmenta a imagem com MediaPipe Selfie Segmentation
        segmentation_results = self.mp_selfie_segmentation.process(image_rgb)

        # A segmentação gera uma máscara com valores de confiança entre 0 e 1
        condition = segmentation_results.segmentation_mask > 0.8  # threshold

        # Cria fundo colorido
        bg_color = (255, 255, 255)  # branco
        bg_image = np.full(image.shape, bg_color, dtype=np.uint8)

        # Combina imagem original e fundo com base na máscara
        final_image = np.where(condition[..., None], image, bg_image)

        return final_image

    # Desenha os pontos das mãos detectados 
    def draw_landmarks(self, image, results, arquivo_csv):
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

                # Adicionando valores
                with open(arquivo_csv, 'a', encoding='utf-8') as f:
                    f.write(f"{angle:.2f}".replace('.', ',') + "\n")
                
                # O resultado do ângulo é exibido
                cv2.putText(image, f'Angulo: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    def run(self):
        contador = 1

        # Descobre o diretório onde está o .exe ou .py
        if getattr(sys, 'frozen', False):
            BASE_DIR = os.path.dirname(sys.executable)
        else:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Pasta de saída
        PASTA_DATAS = os.path.join(BASE_DIR, "datas")
        os.makedirs(PASTA_DATAS, exist_ok=True)

        # Criar o caminho do arquivo
        arquivo_csv = os.path.join(PASTA_DATAS, f"{contador}_angle_between_{self.finger1}_and_{self.finger2}.csv")

        # Cabeçalho
        with open(arquivo_csv, 'w', encoding='utf-8') as f:
            f.write(f"{self.finger1} e {self.finger2}\n")

        # Cria a janela antes de exibir imagem
        window_name = 'Hand Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Define o ícone (Windows)
        icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "icon.ico"))
        set_window_icon(window_name, icon_path)

        while self.cap.isOpened():
            image, results = self.process_frame()
            if image is None:
                break

            self.draw_landmarks(image, results, arquivo_csv)
            cv2.imshow(window_name, image)

            # Fecha no ESC ou no botão X
            if (cv2.waitKey(10) & 0xFF == 27) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                contador += 1
                break

        self.cap.release()
        cv2.destroyAllWindows()
