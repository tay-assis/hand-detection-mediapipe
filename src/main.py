from tkinter import Tk
from hand_detection_gui import HandSelectionGUI
import sys
import os

# Adiciona a pasta src no sys.path, para garantir que modules internos sejam encontrados
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":  
    root = Tk()
    gui = HandSelectionGUI(root)
    root.mainloop()