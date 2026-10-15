"""Render Pillow images using Tk's built-in PPM loader, without Tcl/Tk ABI coupling."""
import tkinter as tk

from PIL import Image


def photo_image(image,master=None):
    if 'A' in image.getbands() or 'transparency' in image.info:
        rgba=image.convert('RGBA')
        background=Image.new('RGBA',rgba.size,'white')
        image=Image.alpha_composite(background,rgba)
    rgb=image.convert('RGB')
    header=f'P6\n{rgb.width} {rgb.height}\n255\n'.encode('ascii')
    return tk.PhotoImage(master=master,data=header+rgb.tobytes(),format='PPM')
