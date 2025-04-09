import math
import time
import tkinter as tk
from tkinter import Canvas, Button
import cv2
import numpy as np
from PIL import Image, ImageTk
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

class GestureRecognitionApp:
    def __init__(self, root, cap, detector, classifier):
        self.root = root
        self.root.title("Sign Conversion App")

        self.cap = cap
        self.detector = detector
        self.classifier = classifier

        self.img_size = 300
        self.offset = 20

        self.canvas = Canvas(root, width=1000, height=800)
        self.canvas.pack()

        self.recognized_text_history = []
        self.recognized_text = ""
        self.start_time = 0

        # TEXT:
        self.text_label = self.canvas.create_text(20, 550, anchor=tk.W, text="TEXT: ", font=("Helvetica", 16), fill="black", tags="text")

        # Clear Text button
        self.clear_button = Button(self.root, text="Clear", command=self.clear_text, height=2, width=10, font=("Helvetica", 14))
        self.clear_button_window = self.canvas.create_window(900, 750, anchor=tk.SE, window=self.clear_button)

        # Backspace button
        self.backspace_button = Button(self.root, text="Backspace", command=self.backspace_text, height=2, width=10, font=("Helvetica", 14))
        self.backspace_button_window = self.canvas.create_window(800, 750, anchor=tk.SE, window=self.backspace_button)

        self.update()

    def update(self):
        global index
        success, img = self.cap.read()
        img_output = img.copy()
        hands, img = self.detector.findHands(img)

        if hands:
            hands = hands[0]
            x, y, w, h = hands['bbox']

            img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255

            x1, y1 = max(0, x - self.offset), max(0, y - self.offset)
            x2, y2 = min(img.shape[1], x + w + self.offset), min(img.shape[0], y + h + self.offset)

            img_crop = img[y1:y2, x1:x2]

            img_crop_shape = img_crop.shape
            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = self.img_size / h
                w_cal = math.ceil(k * w)
                if w_cal > 0:
                    img_resize = cv2.resize(img_crop, (w_cal, self.img_size))
                    img_white[:, (self.img_size - w_cal) // 2:w_cal + (self.img_size - w_cal) // 2] = img_resize
                    prediction, index = self.classifier.getPrediction(img_white, draw=False)

            else:
                k = self.img_size / w
                h_cal = math.ceil(k * h)
                if h_cal > 0:
                    img_resize = cv2.resize(img_crop, (self.img_size, h_cal))
                    img_white[(self.img_size - h_cal) // 2:h_cal + (self.img_size - h_cal) // 2, :] = img_resize
                    prediction, index = self.classifier.getPrediction(img_white, draw=False)

            cv2.putText(img_output, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Update recognized text history
            if labels[index] == "space":
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= 1.0:
                    self.recognized_text_history.append(" ")
                    self.start_time = time.time()
            elif labels[index] == self.recognized_text:
                elapsed_time = time.time() - self.start_time
                if elapsed_time >= 1.5:
                    self.recognized_text_history.append(labels[index])
                    self.start_time = time.time()
            else:
                self.recognized_text = labels[index]
                self.start_time = time.time()

        # Display recognized text on Tkinter canvas
        self.update_text_display()

        img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img_output))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(10, self.update)

    def clear_text(self):
        self.recognized_text_history = []
        self.update_text_display()

    def backspace_text(self):
        if self.recognized_text_history:
            self.recognized_text_history.pop()
            self.update_text_display()

    def update_text_display(self):
        recognized_text_display = "".join(self.recognized_text_history)

        # Replace space word with actual space function
        recognized_text_display = recognized_text_display.replace("space", " ")

        self.canvas.itemconfig("text", text="TEXT: " + recognized_text_display)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/train_model.h5", "Model/labels.txt")

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z", "hello", "yes", "no", "space"]

# Initialize Tkinter
root = tk.Tk()
app = GestureRecognitionApp(root, cap, detector, classifier)
root.mainloop()
