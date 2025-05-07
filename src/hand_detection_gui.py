# hand_selection_gui.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from hand_detection import HandDetection
import os

class HandSelectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Selecione os dedos para calcular o ângulo")

        # Obter caminho absoluto da imagem dentro da classe
        base_dir = os.path.dirname(__file__)
        img_path = os.path.join(base_dir, "images", "png_hand.png")

        # Carregar imagem da mão
        hand_img = Image.open(img_path).resize((400, 400))
        self.hand_photo = ImageTk.PhotoImage(hand_img)

        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.hand_photo)

        # Variáveis para os dois dedos
        self.finger1 = tk.StringVar(value="THUMB_TIP")
        self.finger2 = tk.StringVar(value="INDEX_FINGER_TIP")

        # Lista de opções
        finger_options = [
            ("Polegar", "THUMB_TIP"),
            ("Indicador", "INDEX_FINGER_TIP"),
            ("Médio", "MIDDLE_FINGER_TIP"),
            ("Anelar", "RING_FINGER_TIP"),
            ("Mindinho", "PINKY_TIP"),
        ]

        # Seletor para o primeiro dedo
        ttk.Label(self.root, text="Selecione o primeiro dedo:").pack(anchor=tk.W)
        for text, value in finger_options:
            ttk.Radiobutton(self.root, text=text, variable=self.finger1, value=value).pack(anchor=tk.W)

        # Seletor para o segundo dedo
        ttk.Label(self.root, text="Selecione o segundo dedo:").pack(anchor=tk.W)
        for text, value in finger_options:
            ttk.Radiobutton(self.root, text=text, variable=self.finger2, value=value).pack(anchor=tk.W)

        # Botão de cálculo
        ttk.Button(self.root, text="Calcular Amplitude", command=self.start_detection).pack(pady=10)

    def start_detection(self):
        selected_finger1 = self.finger1.get()
        selected_finger2 = self.finger2.get()
        hand_detection = HandDetection(finger1=selected_finger1, finger2=selected_finger2)
        hand_detection.run()