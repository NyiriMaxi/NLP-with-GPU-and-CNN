import subprocess
import sys
import pkg_resources
import matplotlib
import torch
import numpy
import os
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot
import seaborn 

from dataprocess import updated_dataframe,callbackList

import torch
import torch.nn 
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
import pytesseract
import tkinter 
from tkinter import filedialog
from PIL import Image
import cv2
import easyocr 
import matplotlib.pyplot


"""
def install_requirements():
   
    with open('requirements.txt', 'r') as file:
        packages = file.readlines()
    
    
    required_packages = [pkg.strip() for pkg in packages]
    
    
    for package in required_packages:
        try:
            
            dist = pkg_resources.get_distribution(package)
            print(f"{package} is already installed (version: {dist.version})")
        except pkg_resources.DistributionNotFound:
            
            print(f"{package} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
"""