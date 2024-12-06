import pytesseract as pytes
from PIL import Image
import cv2
import easyocr as easy
import torch
import numpy as np
import matplotlib.pyplot as plt



pytes.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'



#szürkítés

def get_grayscale(image):
    
    if len(image.shape)<=2:
        return image
    else:
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    

#Zajcsökkentés

def remove_noise(image):
    return cv2.medianBlur(image,5)

#küszöbölés

def thresholding(image):
    return cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def dilation(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)


#erode majd utána dilation

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#Canny éldetektáló
def canny(image):
    return cv2.Canny(image, 100, 200)


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)    

def convert_to_rgb(img):
    img=cv2.COLOR_GRAY2BGR
    return img
    
def ReadFromImage(image):
    
    image=np.array(image)



    if image is None:
        print("Kép betöltése sikertelen! Ellenőrizd az elérési utat.")
    else:
        
        
        #A tesseract configuráció beállítása
        #img=cv2.resize(image,(500,400))
        
        
        
        _,binary=thresholding(get_grayscale(image))
        
        custom_config = r'--oem 3 --psm 6'
        plt.imshow(binary)
        plt.show()
        return_text= pytes.image_to_string(binary, config=custom_config,lang="ara+eng+dan+deu+fra+mal+hin+tam+knda+spa+por+rus+ita+swe+nld+tur+grek")
        print (return_text)
        utf8_text = return_text.encode('utf-8').decode('utf-8')
        return utf8_text
        






