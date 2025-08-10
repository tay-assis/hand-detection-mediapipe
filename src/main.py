from tkinter import Tk
from hand_detection_gui import HandSelectionGUI

if __name__ == "__main__":
    root = Tk()
    gui = HandSelectionGUI(root)
    root.mainloop()