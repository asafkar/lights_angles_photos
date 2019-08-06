import tkinter as tk
from tkinter import *
from tkinter import filedialog

from functions import preprocess
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np

class Slider:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.angle = 0
        self.light = 0

        # Load an image using OpenCV
        self.cv_img = np.array(img_list[0][0], dtype=np.uint8)

        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width, no_channels = self.cv_img.shape

        # Create a canvas that can fit the above image
        self.canvas = tk.Canvas(window, width = self.width, height = self.height)
        self.canvas.pack()

        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))

        # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window['padx'] = 5
        self.window['pady'] = 5

        light_scale = tk.Scale(window, from_=0, to=len(img_list)-1, orient=tk.HORIZONTAL, width=8, length=200, command=self.change_light)
        light_scale.pack()

        angle_scale = tk.Scale(window, from_=0, to=len(img_list[0])-1, orient=tk.HORIZONTAL, width=8, length=200, command=self.change_angle)
        angle_scale.pack()

        self.window.mainloop()

    def change_light(self, event):
        self.light = int(event)
        self.cv_img = np.array(img_list[self.light][self.angle], dtype=np.uint8)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def change_angle(self, event):
        self.angle = int(event)
        self.cv_img = np.array(img_list[self.light][self.angle], dtype=np.uint8)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

# Create a window and pass it to the Application object
img_list = preprocess.dict_to_list(preprocess.img_to_arr_led_angle())

Slider(tk.Tk(), "Image Angle and Light Slider")
