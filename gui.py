# Importing libraries
from tkinter import *
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

class DigitClassifier(Frame):

    def __init__(self, parent):
        Frame.__init__(self,parent)
        self.parent = parent
        self.color = "black"
        self.brush_size = 14
        self.setUI()
    
    def setUI(self):
        self.parent.title("Rozpoznawanie cyfry")
        self.pack(fill=BOTH, expand=1)
        self.columnconfigure(6, weight=1)
        self.rowconfigure(2, weight=1)
        self.canv = Canvas(self, bg ="gray")
        self.canv.grid(row=2, column=0, columnspan=7, padx=5, pady=5, sticky=E+W+S+N)
        self.canv.bind("<B1-Motion>", self.draw)
        clear_btn = Button(self, text="Wyczyść", width=30, command=lambda: self.canv.delete("all"), bg="yellow")
        clear_btn.grid(row=3, column=6, sticky=E)

        done_btn = Button(self, text="Sprawdź", width=30, command=lambda: self.save(), bg="yellow")
        done_btn.grid(row=0, column=4)
    
    def save(self):
        self.canv.update()
        ps = self.canv.postscript(colormode="mono")
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save("nasz_rysunek.png")
        digit = DigitClassifier.classify()
        print(digit)
        self.show_digit(digit)
    
    @staticmethod
    def classify():
        clf = load_model("model.h5")
        im = cv2.imread("nasz_rysunek.png",0)
        im2 = cv2.resize(im, (28, 28))
        im = im2.reshape(28, 28, -1)
        im = im.reshape(1, 28, 28, 1)
        im = cv2.bitwise_not(im)
        plt.imshow(im.reshape(28, 28), cmap="Greys")
        result = clf.predict(im)
        digit = np.argmax(result)
        return digit
    
    def show_digit(self, digit):
        text_label= Label(self, text =digit, bg="yellow")
        text_label.grid(row=0,column=6, padx=5, pady=5)

    def draw(self, event):
        self.canv.create_oval(event.x-self.brush_size,
            event.y-self.brush_size,
            event.x+self.brush_size,
            event.y+self.brush_size,
            fill=self.color, outline=self.color)


def mainFunc():
    root = Tk()
    root.geometry("400x400")
    root.resizable(0, 0)
    app = DigitClassifier(root)
    root.mainloop()

if __name__ == '__main__':
    mainFunc()
