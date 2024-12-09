import pytesseract
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from ImageProcessing import ReadFromImage
from model import testing

root = tk.Tk()
root.title("NLP")
root.geometry("500x400")

result_label = tk.Label(root, text="Nyelv előrejelzés: ", font=("Arial", 14))
result_label.pack(pady=20)

def open_image():
   
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        try:       
            img = Image.open(file_path)
                      
            text = ReadFromImage(img)
            text_box.delete(1.0, tk.END)
            text_box.insert(tk.END, text)

            predicted_lang = testing([text])
            result_label.config(text=f"Nyelv előrejelzés: {predicted_lang}")
        except Exception as e:
            text_box.delete(1.0, tk.END)
            text_box.insert(tk.END, f"Hiba történt a kép feldolgozása közben: {str(e)}")

def open_text():
    # Szöveg bevitel
    user_text = tk.simpledialog.askstring("Szöveg beírása", "Írj be egy szöveget:")
    if user_text:
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, user_text)

        # Nyelv előrejelzése a modellel
        predicted_lang = testing([user_text])
        result_label.config(text=f"Nyelv előrejelzés: {predicted_lang}")

open_image_button = tk.Button(root, text="Kép betöltése", command=open_image)
open_image_button.pack(pady=20)

open_text_button = tk.Button(root, text="Szöveg beírása", command=open_text)
open_text_button.pack(pady=20)

text_box = tk.Text(root, wrap=tk.WORD, width=60, height=10)
text_box.pack(padx=20, pady=20)

def runGui():
    root.mainloop()     
