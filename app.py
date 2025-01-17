import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import threading
import win32api
import win32con
import tkinter as tk
from tkinter import font
from tkinter import ttk
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import ImageFont, ImageTk
import PIL.Image
import PredefinedHandGestureComponent
import CustomHandGestureComponent
# from svglib.svglib import svg2rlg
# from reportlab.graphics import renderPM

import time
import asyncio
import tksvg
# import cairosvg

import cv2 as cv
import numpy as np
import mediapipe as mp
import pandas as pd

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from distanceGroup import DistanceGroup

from interfaces import *

from distanceGroup import DistanceGroup

THROTTLE_TIME = 1
last_key_press_time = 0

# global variable
MAINMENU_UI = "mainMenuUI"
PREDEFINED_HG_UI = "predefinedHandGesturesUI"
CUSTOM_HG_UI = "customHandGesturesUI"
TESTING_HG_UI = "testingHandGesturesUI"
PREDEFINED_HG_KEYBOARD_UI = "predefinedHandGesturesKeyboardUI"
PREDEFINED_HG_MOUSE_UI = "predefinedHandGesturesMouseUI"
CUSTOM_HG_KEYBOARD_UI = "customHandGesturesKeyboardUI"
CUSTOM_HG_MOUSE_UI = "customHandGesturesMouseUI"
NEW_CUSTOM_HG_UI = "newCustomHandGestureForKeyboard"
DELETING_LABEL = "Deleting ..."
TRAINING_LABEL = "Training ..."
STARTING_GAME_LABEL = "Initializing webcam ..."

MODEL_SAVE_PATH = "model/keypoint_classifier/keypoint_classifier.keras"
DATASET_PATH = "model/keypoint_classifier/keypoint.csv"
TFLITE_SAVE_PATH = 'model/keypoint_classifier/keypoint_classifier.tflite'
RANDOM_SEED = 42
PREDEFINEDHG_COUNT = 0
NUM_CLASSES = 30
MAX_COLUMN = 3

frameList={}
predefinedKeyboardGesturesList = []
predefinedMouseGesturesList = []
customKeyboardGesturesList = []
customMouseGesturesList = []
isTurboChecked = False

font_path = "VeniteAdoremus.otf"
font_size = 15
button_top_border = '''<svg xmlns="http://www.w3.org/2000/svg" width="603" height="22" viewBox="0 0 603 22" fill="none"><path d="M1 20.5L30 1H570.5L602 20.5" stroke="#37EBFF" stroke-width="2"/></svg>'''
button_bottom_border = '''<svg xmlns="http://www.w3.org/2000/svg" width="603" height="22" viewBox="0 0 603 22" fill="none"><path d="M602 0.999969L573 20.5L32.5 20.5L0.999998 1.00002" stroke="#37EBFF" stroke-width="2"/></svg>'''

def getGameFont():
    pil_font = ImageFont.truetype(font_path, font_size)
    game_font = font.Font(family=pil_font.getname()[0], size=font_size, weight="normal", slant="roman")
    return game_font

# def convert_svg_to_png(svg_file, png_file):
#     drawing = svg2rlg(svg_file)
#     renderPM.drawToFile(drawing, png_file, fmt="PNG")

def getButtonTopBorder():
    top_border_svg = tksvg.SvgImage(data=button_top_border)
    return top_border_svg

def getButtonBottomBorder():
    bottom_border_svg = tksvg.SvgImage(data=button_bottom_border)
    return bottom_border_svg

def getNextSeqOfKeypointCount():
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels_list = [row[0] for row in labels]
        return len(labels_list)
    
def getIndexOfHGFromCSV(gesture):
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels_list = [row[0] for row in labels]
        return labels_list.index(gesture)

def navigateTo(page):
    if page in frameList:
        frame = frameList[page]
        if page in [PREDEFINED_HG_KEYBOARD_UI, PREDEFINED_HG_MOUSE_UI, CUSTOM_HG_KEYBOARD_UI]:
            frame.populatePageElements()
        frame.tkraise()
        print(f"Frame {page} get loaded")
    else:
        print(f"Page {page} not found.")
        
def deleteCustomHG(gesture):
    if gesture in customKeyboardGesturesList:
        customKeyboardGesturesList.remove(gesture)
    else:
        print(f"HG {gesture} not exists.")

def populatePredefinedAndCustomKeyboardGesturesList():
    global predefinedKeyboardGesturesList, customKeyboardGesturesList
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
        encoding='utf-8-sig') as f:
        keypoint_classifier_labels = list(csv.reader(f))
        predefinedKeyboardGesturesList = [
            row[0] for row in keypoint_classifier_labels[:9]
        ]
        print(f"populatePredefinedKeyboardGesturesList() result: {len(predefinedKeyboardGesturesList)}")
        customKeyboardGesturesList = [
            row[0] for row in keypoint_classifier_labels[8:]
        ] 
        print(f"populateCustomKeyboardGesturesList() result: {len(customKeyboardGesturesList)}")
        
# def populateCustomKeyboardgesturesList():
#     global predefinedKeyboardGesturesList, customKeyboardGesturesList
    
        
def produceTrainAndTestDataset():
    X_dataset = np.loadtxt(DATASET_PATH, delimiter=',', dtype='float32', usecols=list(range(1, (21*2)+1)))
    y_dataset = np.loadtxt(DATASET_PATH, delimiter=',', dtype='int32', usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test

def buildModel(model):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def compileModel(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def createCheckpointCallback():
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    return cp_callback, es_callback
        
def trainModelWithCustomHandGesture(frame, text):
    print(f"trainModelWithCustomHandGesture()")
    
    def asyncTask():
        try:
            model = load_model(MODEL_SAVE_PATH)
            model.save(MODEL_SAVE_PATH)
            model = buildModel(model)
            model = compileModel(model)
            X_train, X_test, y_train, y_test = produceTrainAndTestDataset()
            cp_callback, es_callback = createCheckpointCallback()
            model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test), callbacks=[cp_callback, es_callback])
            # val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            # predict_result = model.predict(np.array([X_test[0]]))
            model.save(MODEL_SAVE_PATH, include_optimizer=False)    
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_quantized_model = converter.convert()

            open(TFLITE_SAVE_PATH, 'wb').write(tflite_quantized_model)      
        finally:
            frame.after(0, hide_loading_popup, loading_window)
    
    loading_window, spinner = show_loading_popup(frame, text)
    threading.Thread(target=asyncTask, daemon=True).start()
            

def show_loading_popup(root, text):
    # Create a new top-level window to display the loading spinner and label
    loading_window = tk.Toplevel(root)
    loading_window.title("Loading")
    loading_window.geometry("200x100")  # Set the size of the popup window
    loading_window.configure(bg="white")

    # Make the window stay on top of the main window
    loading_window.attributes("-topmost", True)
    loading_window.grab_set()  # Make the window modal (blocks interaction with other windows)

    # Disable the close button of the popup window
    loading_window.protocol("WM_DELETE_WINDOW", lambda: None)

    # Center the loading window on the parent window (root)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - 50)  # Adjust the position to center
    position_left = int(screen_width / 2 - 100)
    loading_window.geometry(f"200x100+{position_left}+{position_top}")

    # Create a label above the spinner
    label = ttk.Label(loading_window, text=text, font=("Arial", 14), foreground="red", background="white")
    label.pack(pady=10)

    # Create the spinner (progress bar)
    spinner = ttk.Progressbar(loading_window, mode="indeterminate")
    spinner.pack(pady=10)
    spinner.start()

    return loading_window, spinner

def hide_loading_popup(loading_window):
    # Close the loading window
    loading_window.destroy()

def addHandGestureToCSV(newHG):
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([newHG])
        print(f"Added new value to keypoint CSV: {newHG}")

def addNewCustomGesture(frame, newHG_name):
    enteredHGName = newHG_name.get()
    print(f"addNewCustomGesture(), argument: {enteredHGName}")
    if enteredHGName != "":
        # customHG = CustomHandGestureObject(enteredHGName)
        customKeyboardGesturesList.append(enteredHGName)
        print(f"customKeyboardGesturesList count: {len(customKeyboardGesturesList)}")
        newHG_name.delete(0, tk.END)
        addHandGestureToCSV(enteredHGName)
        trainModelWithCustomHandGesture(frame, TRAINING_LABEL)
    navigateTo(CUSTOM_HG_KEYBOARD_UI) 

def deleteCustomGesture(customHGName, frame):
    def task():
        try:
            print(f"deleteCustomGesture(), argument: {customHGName}")
            global customKeyboardGesturesList
            customKeyboardGesturesList = [name for name in customKeyboardGesturesList if name != customHGName]
            print(f"deleteCustomGesture() updated customKeyboardGesturesList: {len(customKeyboardGesturesList)}")
            customHGIndex = deleteHandGestureFromCSV(customHGName)
            updateKeypointCSV(customHGIndex)
            trainModelWithCustomHandGesture(frame, DELETING_LABEL)
            frame.populatePageElements()
        finally:
            # Hide the spinner when the task is done
            frame.after(0, hide_loading_popup, loading_window)

    # Show the loading spinner
    loading_window, spinner = show_loading_popup(frame, DELETING_LABEL)

    # Run the task in a separate thread
    threading.Thread(target=task, daemon=True).start()
    
def deleteHandGestureFromCSV(gesture):
    print(f"deleteHandGestureFromCSV()")
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', mode='r', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
        
    # print(f"rows: {rows}")
    index = 0
    for rowIndex, row in enumerate(rows):
        if gesture in row:
            index = rowIndex
    updated_rows = [row for row in rows if row and row[0] != gesture]
    
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)
    
    print(f"Deleted value from keypoint csv: {gesture}")
    return index

def updateKeypointCSV(index):
    print(f"updateKeypointCSV()")
    dataframe = pd.read_csv(DATASET_PATH, header=None, engine="python", on_bad_lines="skip")
    updatedDataframe = dataframe[dataframe[0] != index].reset_index(drop=True)
    updatedDataframe[0] = updatedDataframe[0].where(updatedDataframe[0] < index, updatedDataframe[0]-1)
    updatedDataframe.to_csv(DATASET_PATH, index=False, header=False)

# def initializeUI():

class CustomHandGestureObject:
    def __init__(self, name):
        self.name = name
        
class Root(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Playing Games with Bare Hands")
        self.geometry("500x500")
        self.configure(bg="#f0f0f0")
        
        mainFrame = Frame(self)
        mainFrame.pack(fill="both", expand=True)
        
        mainFrame.grid_rowconfigure(0, weight=1)
        mainFrame.grid_columnconfigure(0, weight=1)
        
        populatePredefinedAndCustomKeyboardGesturesList()
        # self.load_game_font()
        # self.frames = {}
        
        for page in (MainMenuUI, PredefinedHandGesturesUI, CustomHandGesturesUI, PredefinedHandGesturesKeyboardUI, PredefinedHandGesturesMouseUI, CustomHandGesturesKeyboardUI, NewCustomHandGestureForKeyboard):
            print(f"Initializing frame for {page.getIdentity()}")
            frame = page(mainFrame, self)
            # if page.getIdentity() in [PREDEFINED_HG_KEYBOARD_UI, PREDEFINED_HG_MOUSE_UI, CUSTOM_HG_KEYBOARD_UI]:
            #     page.populatePageElements(frame)
            frameList[page.getIdentity()] = frame
            frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
        print("frames count: {0}".format(str(len(frameList))))
            
        for x in frameList:
            print("frames item: {0}".format(x))
        
        navigateTo(MAINMENU_UI)
    
    # def load_game_font(self):
    #     if os.path.exists(font_path):
    #         self.tk.call("font", "create", "venite_font", "-family", "Venite Adoremus", "-size", 20, "-weight", "normal")
    #         self.tk.call("font", "configure", "venite_font", "-file", font_path)
    #     else:
    #         print("Font file for game font not found. Please provide correct file.")

def loadWallpaper(root):
    wallpaper = PIL.Image.open("GWBHWallpaper.jpg")
    root.bg_image = ImageTk.PhotoImage(wallpaper.resize((root.winfo_screenwidth(), root.winfo_screenheight())))
    bg_label = Label(root, image=root.bg_image)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    
def loadProductName(root):
    icon = PIL.Image.open("productIcon.png")
    root.productIcon = ImageTk.PhotoImage(icon)
    root.grid_columnconfigure(0, weight=1)
    productName = Label(root, text="GAMING WITH BARE HANDS", font=("Venite Adoremus", 30, 'bold'), fg="#FFF", bg="black", justify="center", image=root.productIcon, compound="left")
    productName.grid(row=0, column=0, columnspan=10, pady=10, sticky="nsew")

class MainMenuUI(ttk.Frame):
    global isTurboChecked
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame)
        self.root = root
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=mainMenuTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        predefinedHandGesturesBtn =buildButton(self, "Predefined Hand Gestures", navigateTo, PREDEFINED_HG_UI)
        frameButton(self, predefinedHandGesturesBtn)

        customHandGesturesBtn = buildButton(self, "Custom Hand Gestures", navigateTo, CUSTOM_HG_UI)
        frameButton(self, customHandGesturesBtn)
        
        testingBtn = buildButton(self, "Test Hand Gestures", initiateWebCam, False)
        frameButton(self, testingBtn)
        
        startGameBtn = buildButton(self, "Start Game", initiateWebCam, True)
        frameButton(self, startGameBtn)
        
        label = Label(self, text="Turbo:", font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", anchor="w", justify="center")  # Align the text to the left
        label.grid(padx=20, pady=20)

        self.turboChecked = tk.BooleanVar()
        self.turboChecked.trace_add("write", self.onTurboChecked)
        checkbox = ttk.Checkbutton(self, variable=self.turboChecked)
        # isTurboChecked = self.checkbox_var.get()
        print(f"checkbox_var: {self.turboChecked.get()}")
        checkbox.grid(padx=20, pady=20)
    
    def onTurboChecked(self, *args):
        global isTurboChecked
        isTurboChecked = self.turboChecked.get()
        print(f"isTurboChecked: {isTurboChecked}")
        
    def getIdentity():
        return MAINMENU_UI
        
class PredefinedHandGesturesUI(ttk.Frame):
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=predefinedHGTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        keyboardBtn = buildButton(self, "Keyboard", navigateTo, PREDEFINED_HG_KEYBOARD_UI)
        frameButton(self, keyboardBtn)
        mouseBtn = buildButton(self, "Mouse", navigateTo, PREDEFINED_HG_MOUSE_UI)
        frameButton(self, mouseBtn)
        doneBtn = buildDoneButton(self, "Done", navigateTo, MAINMENU_UI)
        frameButton(self, doneBtn)
        
        # keyboardBtn.pack(padx=20, pady=20)
        # mouseBtn.pack(padx=20, pady=20)
        # doneBtn.pack(padx=20, pady=20)
        
    def getIdentity():
        return PREDEFINED_HG_UI
        

class CustomHandGesturesUI(ttk.Frame):
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=customHGTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        keyboardBtn = buildButton(self, "Keyboard", navigateTo, CUSTOM_HG_KEYBOARD_UI)
        frameButton(self, keyboardBtn)
        # mouseBtn = buildButton(self, "Mouse", navigateTo, "TOCHANGE")
        # frameButton(self, mouseBtn)
        doneBtn = buildDoneButton(self, "Done", navigateTo, MAINMENU_UI)
        frameButton(self, doneBtn)
        
        # keyboardBtn.pack(padx=20, pady=20)
        # mouseBtn.pack(padx=20, pady=20)
        # doneBtn.pack(padx=20, pady=20)
        
    def getIdentity():
        return CUSTOM_HG_UI  

class PredefinedHandGesturesComponent(ttk.Frame):
    def __init__(self, mainFrame, gesture_name, action, has_value_var):
        super().__init__(mainFrame)
        
        # Label for hand gesture (row 0, column 0, span entire row)
        self.label = Label(self, text=gesture_name,  font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black")
        self.label.grid(row=0, column=0, columnspan=2, pady=5, sticky="n")
        
        # Button to record (row 1, column 0)
        self.record_button = Button(self, text="Click to Record", command=lambda g=gesture_name, l=self.label: action(g, l))
        self.record_button.grid(row=1, column=0, pady=5, sticky="n")
        
        # Checkbox to show recorded value status (row 1, column 1)
        self.checkbox = Checkbutton(self, variable=has_value_var, state="disabled")
        self.checkbox.grid(row=1, column=1, padx=5, sticky="w")
        
        # Configure grid column weights within the component
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        def on_enter(event, b=self.record_button):
            b.config(style='Hover.TButton')
                
        def on_leave(event, b=self.record_button):
            b.config(style='TButton')
                    
        self.record_button.bind("<Enter>", on_enter)
        self.record_button.bind("<Leave>", on_leave)

class PredefinedHandGesturesKeyboardUI(ttk.Frame):
    global predefinedKeyboardGesturesList
        
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        # self.gesture_to_key = {}
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=predefinedHGTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))

        #populate predefined keyboard gestures
        # populatePredefinedKeyboardGesturesList(predefinedKeyboardGesturesList)
        self.populatePageElements()
        
        
    def populatePageElements(self):
        print(f"PredefinedHandGesturesKeyboardUI populatePageElements()")
        print(f"PredefinedHandGesturesKeyboardUI predefinedKeyboardGesturesList: {len(predefinedKeyboardGesturesList)}")
        
        for widget in self.winfo_children():
            widget.destroy()
            
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=predefinedHGTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10, columnspan=MAX_COLUMN)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        
        for col in range(MAX_COLUMN):
            self.grid_columnconfigure(col, weight=1, uniform="col")
            
        total_rows = (len(predefinedKeyboardGesturesList) + MAX_COLUMN - 1) // MAX_COLUMN
        for row in range(total_rows + 2):
            self.grid_rowconfigure(row, weight=1, uniform="row")
            
        # Add labels and buttons for each gesture
        for idx, gesture in enumerate(predefinedKeyboardGesturesList):
            if idx < len(predefinedKeyboardGesturesList)-1:

                componentRow = idx // MAX_COLUMN
                componentColumn = idx % MAX_COLUMN
                
                hgComponent = PredefinedHandGestureComponent.HandGestureComponent(self, label_text=gesture, button_command=self.enable_recording)
                hgComponent.grid(row=componentRow+2, column=componentColumn, ipadx=50, ipady=50)
                
        # for row in range((len(predefinedKeyboardGesturesList) + 2) // 3):  # Total rows
        #     mainFrame.grid_rowconfigure(row, weight=1)
        
        # Add a proceed button
        # proceed_button = ttk.Button(self, text="Proceed", command=self.proceed, style='TButton')
        proceed_button = buildDoneButton(self, "Done", navigateTo, PREDEFINED_HG_UI)
        proceed_button.grid(row=len(predefinedKeyboardGesturesList) + 2, column=0, columnspan=MAX_COLUMN, pady=20)
        
    def record_key(self, event, gesture, label):
        # Store the hexadecimal code of the key using ord()
        vk_code = win32api.VkKeyScan(event.char)
        key_name = event.keysym
        key_mapping[gesture] = vk_code
        label.config(text=f"{gesture}: {key_name}", foreground="green")
        self.root.unbind("<Key>")

    # Define a function to enable key recording
    def enable_recording(self, gesture, label):
        label.config(text=f"{gesture}: Press any key...", foreground="orange")
        self.root.bind("<Key>", lambda event: self.record_key(event, gesture, label))

    # Define a function that will be called when the user clicks "Proceed"
    # def proceed(self):
    #     print("Gesture to key mapping:")
    #     # for gesture, hex_code in self.gesture_to_key.items():
    #     for gesture,hex_code in key_mapping.items():
    #         print(f"{gesture}: {hex_code}")
    #     # Here you can add any additional actions for when the proceed button is pressed
    #     self.root.quit()
        
    def getIdentity():
        return PREDEFINED_HG_KEYBOARD_UI
    
class PredefinedHandGesturesMouseUI(ttk.Frame):
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        self.gesture_to_key = {}
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=predefinedHGTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid()
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        self.populatePageElements()

    def populatePageElements(self):
        print(f"PredefinedHandGesturesMouseUI populatePageElements()")
        print(f"PredefinedHandGesturesMouseUI predefinedMouseGesturesList: {len(predefinedMouseGesturesList)}")
        
        for widget in self.winfo_children():
            widget.destroy()
            
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=predefinedHGTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10, columnspan=MAX_COLUMN)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        
        for col in range(MAX_COLUMN):
            self.grid_columnconfigure(col, weight=1, uniform="col")
                
        total_rows = (len(predefinedMouseGesturesList) + MAX_COLUMN - 1) // MAX_COLUMN
        for row in range(total_rows+2):
            self.grid_rowconfigure(row, weight=1, uniform="row")
        # Add labels and buttons for each gesture
        for idx, gesture in enumerate(predefinedMouseGesturesList):
            
            componentRow = idx // MAX_COLUMN
            componentColumn = idx % MAX_COLUMN
            
            hgComponent = PredefinedHandGestureComponent.HandGestureComponent(self, label_text=gesture, button_command=self.enable_recording)
            hgComponent.grid(row=componentRow+2, column=componentColumn, ipadx=50, ipady=50)
            
            # label = ttk.Label(self, text=f"{gesture}: Not assigned", font=("Venite Adoremus", 10, 'bold'), foreground="red")
            # label.grid(row=idx + 1, column=0, padx=5, pady=10, sticky=tk.W)
            
            # button = ttk.Button(self, text="Click to record", command=lambda g=gesture, l=label: self.enable_recording(g, l))
            # button.grid(row=idx + 1, column=1, padx=10, pady=10)
            
            # # Add hover effect to buttons
            # def on_enter(event, b=button):
            #     b.config(style='Hover.TButton')
            
            # def on_leave(event, b=button):
            #     b.config(style='TButton')
                
            # button.bind("<Enter>", on_enter)
            # button.bind("<Leave>", on_leave)
            
        
        # Add a proceed button
        # proceed_button = ttk.Button(self, text="Proceed", command=self.proceed, style='TButton')
        proceed_button = buildDoneButton(self, "Done", navigateTo, PREDEFINED_HG_UI)
        proceed_button.grid(row=len(predefinedMouseGesturesList) + 2, column=0, columnspan=MAX_COLUMN, pady=20)
        
    def record_key(self, event, gesture, label):
        # Store the hexadecimal code of the key using ord()
        vk_code = win32api.VkKeyScan(event.char)
        key_name = event.keysym
        mouse_mapping[gesture] = vk_code
        label.config(text=f"{gesture}: {key_name}", foreground="green")
        self.root.unbind("<Key>")

    # Define a function to enable key recording
    def enable_recording(self, gesture, label):
        label.config(text=f"{gesture}: Press any key...", foreground="orange")
        self.root.bind("<Key>", lambda event: self.record_key(event, gesture, label))

    # Define a function that will be called when the user clicks "Proceed"
    # def proceed(self):
    #     print("Gesture to key mapping:")
    #     for gesture, hex_code in self.gesture_to_key.items():
    #         print(f"{gesture}: {hex_code}")
    #     # Here you can add any additional actions for when the proceed button is pressed
    #     self.root.quit()
        
    def getIdentity():
        return PREDEFINED_HG_MOUSE_UI
    
    
class CustomHandGesturesKeyboardUI(ttk.Frame):
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        # self.gesture_to_key = {}
        # self.custom_key_mapping = {}
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text="CUSTOM HAND GESTURES", font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        
        print(f"customHandGesturesKeyboardUI loaded. customKeyboardGesturesList has {len(customKeyboardGesturesList)} items")
        self.populatePageElements()
        
    def populatePageElements(self):
        print(f"CustomHandGesturesKeyboardUI populatePageElements()")
        print(f"CustomHandGesturesKeyboardUI customKeyboardGesturesList: {len(customKeyboardGesturesList)}")
        
        for widget in self.winfo_children():
            widget.destroy()
            
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text="CUSTOM HAND GESTURES", font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10, columnspan=MAX_COLUMN)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        
        for col in range(MAX_COLUMN):
            self.grid_columnconfigure(col, weight=1, uniform="col")
            
        total_rows = (len(predefinedKeyboardGesturesList) + MAX_COLUMN - 1) // MAX_COLUMN
        for row in range(total_rows + 2):
            self.grid_rowconfigure(row, weight=1, uniform="row")
        
        # Add labels and buttons for each gesture
        if customKeyboardGesturesList:
            for idx, customHGName in enumerate(customKeyboardGesturesList):
                
                componentRow = idx // MAX_COLUMN
                componentColumn = idx % MAX_COLUMN
                
                hgComponent = CustomHandGestureComponent.HandGestureComponent(self, label_text=customHGName, button_command=self.enable_recording, delete_button_command=deleteCustomGesture)
                hgComponent.grid(row=componentRow+2, column=componentColumn, ipadx=50, ipady=50)
        
        add_gesture_button = buildButton(self, "Add New Hand Gesture", navigateTo, NEW_CUSTOM_HG_UI)
        add_gesture_button.grid(row=len(customKeyboardGesturesList)+2, column=0, columnspan=MAX_COLUMN, pady=20)
        
        # Add a proceed button
        # proceed_button = ttk.Button(self, text="Proceed", command=self.proceed, style='TButton')
        proceed_button = buildDoneButton(self, "Done", navigateTo, CUSTOM_HG_UI)
        proceed_button.grid(row=len(customKeyboardGesturesList) +3, column=0, columnspan=MAX_COLUMN, pady=20)
        
    def record_key(self, event, gesture, label):
        # Store the hexadecimal code of the key using ord()
        vk_code = win32api.VkKeyScan(event.char)
        key_name = event.keysym
        key_mapping[gesture] = vk_code
        label.config(text=f"{gesture}: {key_name}", foreground="green")
        self.root.unbind("<Key>")

    # Define a function to enable key recording
    def enable_recording(self, gesture, label):
        label.config(text=f"{gesture}: Press any key...", foreground="orange")
        self.root.bind("<Key>", lambda event: self.record_key(event, gesture, label))
        
    def getIdentity():
        return CUSTOM_HG_KEYBOARD_UI

class NewCustomHandGestureForKeyboard(ttk.Frame):
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        self.gesture_to_key = {}
        self.custom_key_mapping = {}
        self.newHGName = ''
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)
        
        loadWallpaper(self)
        loadProductName(self)
        
        newHG_label = Label(self, text='Hand Gesture Name', font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black", justify="center")
        newHG_label.grid(row=1, column=0, padx=(0, 10), pady=5, sticky="e")
        newHG_entry = Entry(self, textvariable=self.newHGName, font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black", justify="center")
        newHG_entry.grid(row=1, column=1, pady=5, sticky="w")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        
        keypointCount = getNextSeqOfKeypointCount()
        print(f"Total count in keypoints csv: {keypointCount}")
                
        add_gesture_button = buildButton(self, "Click to record", initiateWebCam, False)
        add_gesture_button.grid(row=2, column=0, columnspan=2, pady=20)
        
        # done_button = buildDoneButton(self, "Done", addNewCustomGesture, newHG_entry)
        done_button = Button(
            self,
            text="Done",
            command= lambda: addNewCustomGesture(self, newHG_entry),
            activebackground="blue",
            activeforeground="white",
            anchor="center",
            bd=3,
            bg="black",
            cursor="hand2",
            foreground="#37EBFF",
            fg="#37EBFF",
            font=getGameFont(),
            height=2,
            highlightbackground="black",
            highlightcolor="green",
            highlightthickness=2,
            justify="center",
            overrelief="raised",
            padx=10,
            pady=5,
            width=25,
            wraplength=0
        )
        done_button.grid(row=4, column=0, columnspan=2, pady=20)
        
    def getIdentity():
        return NEW_CUSTOM_HG_UI
        
class TestingHGUI(ttk.Frame):
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        
        title = Label(self, text=customHGTitle, font=titleSize)
        title.grid()
        
        # keyboardBtn = buildButton(self, "Keyboard", )
        # mouseBtn = buildButton(self, "Mouse", )
        # doneBtn = buildButton(self, "Done", , "green")
        
        # keyboardBtn.pack(padx=20, pady=20)
        # mouseBtn.pack(padx=20, pady=20)
        # doneBtn.pack(padx=20, pady=20)
        
    def getIdentity():
        return TESTING_HG_UI

def frameButton(root, button):
    # cairosvg.svg2png(bytestring=button_top_border, write_to="top_border.png")
    # cairosvg.svg2png(bytestring=button_bottom_border, write_to="bottom_border.png")
    top_border = tksvg.SvgImage(data=button_top_border)
    top_border_label = Label(root, image=top_border, borderwidth=0)
    # top_border_image = Image.open("top_border.png")
    # bottom_border_image = Image.open("bottom_border.png")
    # top_tk_image = ImageTk.PhotoImage(top_border_image)
    # bottom_tk_image = ImageTk.PhotoImage(bottom_border_image)
    # top_border_label = Label(root, image=top_tk_image, borderwidth=0)
    # bottom_border_label = Label(root, image=bottom_tk_image, borderwidth=0)
    bottom_border = tksvg.SvgImage(data=button_bottom_border)
    bottom_border_label = Label(root, image=bottom_border, borderwidth=0)
    # top_border_label.pack()
    button.grid(padx=20, pady=20)
    # bottom_border_label.pack()

        
def buildButton(frame, text, actionFunc, pageName):
    button = Button(
        frame,
        text=text,
        command= lambda: actionFunc(pageName) if pageName is not None else actionFunc(),
        activebackground="blue",
        activeforeground="white",
        anchor="center",
        bd=3,
        bg="black",
        cursor="hand2",
        foreground="#37EBFF",
        fg="#37EBFF",
        font=getGameFont(),
        height=2,
        highlightbackground="black",
        highlightcolor="green",
        highlightthickness=2,
        justify="center",
        overrelief="raised",
        padx=10,
        pady=5,
        width=25,
        wraplength=0
    )
    
    return button

def buildDoneButton(frame, text, actionFunc, pageName):
    button = Button(
        frame,
        text=text,
        command= lambda: actionFunc(pageName) if pageName is not None else actionFunc(),
        activebackground="blue",
        activeforeground="white",
        anchor="center",
        bd=3,
        bg="black",
        cursor="hand2",
        foreground="#37EBFF",
        fg="#37EBFF",
        font=getGameFont(),
        height=2,
        highlightbackground="black",
        highlightcolor="green",
        highlightthickness=2,
        justify="center",
        overrelief="raised",
        padx=10,
        pady=5,
        width=10,
        wraplength=0
    )
    
    return button

def buildNavigationButton(frame, text, actionFunc, pageName):
    
    button = Button(
        frame,
        text=text,
        command= lambda: actionFunc(pageName) if pageName is not None else actionFunc(),
        activebackground="blue",
        activeforeground="white",
        anchor="center",
        bd=3,
        bg="lightgray",
        cursor="hand2",
        foreground="black",
        fg="black",
        font=("Arial", 12),
        height=2,
        highlightbackground="black",
        highlightcolor="green",
        highlightthickness=2,
        justify="center",
        overrelief="raised",
        padx=10,
        pady=5,
        width=15,
        wraplength=100
    )
    
    return button

def buildButtonWithColor(frame, text, actionFunc, pageName, color):
    
    button = Button(
        frame,
        text=text,
        command= lambda: actionFunc(pageName),
        activebackground="blue",
        activeforeground=color,
        anchor="center",
        bd=3,
        bg="lightgray",
        cursor="hand2",
        foreground="black",
        fg="black",
        font=("Arial", 12),
        height=2,
        highlightbackground="black",
        highlightcolor="green",
        highlightthickness=2,
        justify="center",
        overrelief="raised",
        padx=10,
        pady=5,
        width=15,
        wraplength=100
    )
    
    return button

    #  ===================================================================================================================== User Interfaces ===========================================================================================================

# Default mapping if user doesnt record keybinds, DO NOT CHANGE

key_mapping = {
    "Up": 0x26,       # VK_UP (Arrow Up)
    "Down": 0x28,     # VK_DOWN (Arrow Down)
    "Left": 0x25,     # VK_LEFT (Arrow Left)
    "Right": 0x27,    # VK_RIGHT (Arrow Right)
    "Start": 0x0D,    # VK_RETURN (Enter/Start)
    "Select": 0x20,   # VK_SPACE (Space/Select)
    "A": 0x41,        # A key
    "B": 0x42         # B key
}
mouse_mapping = {
    "LeftClick": 0x01,   # Left mouse button
    "RightClick": 0x02,  # Right mouse button
    "MiddleClick": 0x04  # Middle mouse button
}


async def press_key_turbo(key_name):
    global last_key_press_time
    current_time = time.time()

    # Ensure the key_name is valid
    if key_name not in key_mapping:
        print(f"Invalid key name: {key_name}")
        return

    # Get the hex key code for the provided key name
    hex_key_code = key_mapping[key_name]

    # Check if the time since the last key press is greater than the throttle time.
    # if current_time - last_key_press_time >= THROTTLE_TIME:
        # Press the key down
    win32api.keybd_event(hex_key_code, 0, 0, 0)
    # time.sleep(0.05)  # Short delay to ensure the key press is registered
        
        # Simulate releasing the key
    # win32api.keybd_event(hex_key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    last_key_press_time = current_time
        
    print(f"key pressed.")


async def press_key_throttled(key_name):
    global last_key_press_time
    current_time = time.time()

    # Ensure the key_name is valid
    if key_name not in key_mapping:
        print(f"Invalid key name: {key_name}")
        return

    # Get the hex key code for the provided key name
    hex_key_code = key_mapping[key_name]

    # Check if the time since the last key press is greater than the throttle time.
    # if current_time - last_key_press_time >= THROTTLE_TIME:
        # Press the key down
    win32api.keybd_event(hex_key_code, 0, 0, 0)
    time.sleep(0.2)  # Short delay to ensure the key press is registered
        
        # Simulate releasing the key
    win32api.keybd_event(hex_key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    last_key_press_time = current_time
        
    print(f"key pressed.")
        
def initiateWebCam(isGameStart):
    
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    # KeypointClassifier mainly run through all the data from keypoint_classifier.tflite to predict what is the class of the input data, output will be choosing the class with highest probability.
    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    # point_history store most recent 16 points
    point_history = deque(maxlen=history_length)
    #  finger_gesture_history store most recent 16 gestures 
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0
    
    thumbDownCount = 0
    thumbUpCount = 0
    closeCount = 0
    okayCount = 0

    while True:
        fps = cvFpsCalc.get()
        
        key = cv.waitKey(10)
        if key == 27:  # ESC
            
            break
        number, mode = select_mode(key, mode)

        # means if the camera capture something, then rest is True. Else rest is False.
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # multi_hand_landmarks contains one entry if only one hand detected, each entry has 21 landmarks which represents the hand gesture.
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Convert each landmark provided to (x, y) format, then normalise coordinates to pixels by multiplying widths and heights
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # pre_process_landmark() will normalise a list of landmarks
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # pre_process_point_history will normalise a list of point history
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # keypoint_classifier take in a list of normalised landmarks as input, then Tensorflow lite model will run inference to classify hand gestures based on input, then produce a list of probabilities as output. The hand gesture with highest probability will be the predicted hand gesture.
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                
                
                gesture_text = keypoint_classifier_labels[hand_sign_id]

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    gesture_text,
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                if isGameStart:
                    if isTurboChecked:
                        asyncio.run(press_key_turbo(gesture_text))
                        print(f"Turbo is on, press_key_turbo is running")
                    else:
                        asyncio.run(press_key_throttled(gesture_text))
                        print(f"Turbo is off, press_key_throttled is running")
                # asyncio.run(press_key_throttled("A"))
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('HGR To Play Games', debug_image)

    cap.release()
    cv.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.7)

    args = parser.parse_args()

    return args


def main():
    # Create a mapping for the gestures and keypresses
    gesture_to_key = {}
    
    thumbUpAlgo = DistanceGroup(5000)
    thumbDownAlgo = DistanceGroup(5000)
    closeAlgo = DistanceGroup(5000)
    okayAlgo = DistanceGroup(5000)

    # Define the list of gestures
    gestures = ["Middle Finger", "Thumb Up", "Thumb Down", "Peace Sign", "OK Sign", "Fist", "Close"]
    
    # Create a function to handle keypress events
    def record_key(event, gesture, label):
        # Store the hexadecimal code of the key using ord()
        vk_code = win32api.VkKeyScan(event.char)
        gesture_to_key[gesture] = vk_code
        label.config(text=f"{gesture}: {vk_code}", foreground="green")
        root.unbind("<Key>")

    # Define a function to enable key recording
    def enable_recording(gesture, label):
        label.config(text=f"{gesture}: Press any key...", foreground="orange")
        root.bind("<Key>", lambda event: record_key(event, gesture, label))

    # Define a function that will be called when the user clicks "Proceed"
    # def proceed():
    #     print("Gesture to key mapping:")
    #     for gesture, hex_code in gesture_to_key.items():
    #         print(f"{gesture}: {hex_code}")
    #     # Here you can add any additional actions for when the proceed button is pressed
    #     root.quit()

    root = Root()

    # Run the application
    root.mainloop()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 32:   # space bar
        mode = 3
        number = getNextSeqOfKeypointCount()
        print(f"Spacebar pressed. Number of gesture in keypoint.csv: {number}")
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - 
                                        base_y) / image_height

    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    if mode == 3:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([getNextSeqOfKeypointCount(), *landmark_list])
            print(f"Write into keypoint.csv for row {getNextSeqOfKeypointCount()+1}")
        


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # index
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # middle
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # ring
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # pinky
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        if index == 0:  # wrist
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # wrist
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # thumb base
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # thumb joint 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # thumb tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # index base
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # index joint 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # index joint 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # index tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 255, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # middle base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # middle joint 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # middle joint 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # middle tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 0, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # ring base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 100),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # ring joint 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 100),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # ring joint 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 100),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # ring tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 0, 100),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # pinky base
            cv.circle(image, (landmark[0], landmark[1]), 5, (69, 100, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # pinky joint 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (69, 100, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # pinky joint 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (69, 100, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # pinky tip
            cv.circle(image, (landmark[0], landmark[1]), 8, (69, 100, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()

# def initialistPages(mainFrame, root):
#     frames = {}
    
#     for page in (mainMenuUI, predefinedHandGesturesUI, customHandGesturesUI):
#         frame = page(mainFrame, root)
#         frames[page] = frame
    
#     return frames