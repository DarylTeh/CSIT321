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
from tkinter import messagebox
from tkinter import StringVar
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import ImageFont, ImageTk
import PIL.Image
import PredefinedHandGestureComponent
import CustomHandGestureComponent

import time
import asyncio
import tksvg
import cv2 as cv
import numpy as np
import mediapipe as mp
import pandas as pd
import os

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from distanceGroup import DistanceGroup

from interfaces import *

from distanceGroup import DistanceGroup
from pathlib import Path
import shutil

# global variable
MAINMENU_UI = "mainMenuUI"
OPTION_UI = "optionUI"
PREDEFINED_HG_UI = "predefinedHandGesturesUI"
CUSTOM_HG_UI = "customHandGesturesUI"
TESTING_HG_UI = "testingHandGesturesUI"
PREDEFINED_HG_KEYBOARD_UI = "predefinedHandGesturesKeyboardUI"
PREDEFINED_HG_MOUSE_UI = "predefinedHandGesturesMouseUI"
CUSTOM_HG_KEYBOARD_UI = "customHandGesturesKeyboardUI"
CUSTOM_HG_MOUSE_UI = "customHandGesturesMouseUI"
NEW_CUSTOM_HG_UI = "newCustomHandGestureForKeyboard"
NEW_PROFILE_UI = "newProfileComponent"

DELETING_LABEL = "Deleting ..."
TRAINING_LABEL = "Training ..."
INITIATE_WEBCAM_LABEL = "Initializing webcam ..."

MODEL_SAVE_PATH = "model/keypoint_classifier/keypoint_classifier.keras"
DATASET_PATH = "model/keypoint_classifier/"
DATASET_FILENAME = 'keypoint'
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
key_mapping = {}
keystroke_binding = {}
profileList = []
allGesturesPreview = {}


font_path = "VeniteAdoremus.otf"
font_size = 15
button_top_border = '''<svg xmlns="http://www.w3.org/2000/svg" width="603" height="22" viewBox="0 0 603 22" fill="none"><path d="M1 20.5L30 1H570.5L602 20.5" stroke="#37EBFF" stroke-width="2"/></svg>'''
button_bottom_border = '''<svg xmlns="http://www.w3.org/2000/svg" width="603" height="22" viewBox="0 0 603 22" fill="none"><path d="M602 0.999969L573 20.5L32.5 20.5L0.999998 1.00002" stroke="#37EBFF" stroke-width="2"/></svg>'''
KEYSTROKE_BINDING_FILEPATH = 'model/keypoint_classifier/'
KEYSTROKE_BINDING_FILENAME = 'keypoint_classifier_keystroke_binding'
FILEEXT = '.csv'
KEYPOINT_LABEL_FILEPATH = 'model/keypoint_classifier/'
KEYPOINT_LABEL_FILENAME = 'keypoint_classifier_label'
isFirstRender = True
SELECTED_PROFILE = "default"
CURRENT_PROFILE_KEYSTROKE_BINDING_PATH = ''
CURRENT_PROFILE_KEYPOINT_LABEL_PATH = ''
CURRENT_PROFILE_KEYPOINT_PATH = ''
TEMPLATE_KEYSTROKE_BINDING_FILEPATH = 'model/keypoint_classifier/keypoint_classifier_keystroke_binding_template.csv'
TEMPLATE_KEYPOINT_LABEL_FILEPATH = 'model/keypoint_classifier/keypoint_classifier_label_template.csv'
TEMPLATE_DATASET_FILEPATH = 'model/keypoint_classifier/keypoint.csv'
PROFILES_FILEPATH = 'profiles.csv'

# After select another profile, update current profile, check if the csv files for current selected profile exists, then load all data from respective csv files
# this method will just load the data from respoective csv files.
def reloadProfileDependantData():
    print(f"reloadProfileDependantData()")
    global predefinedKeyboardGesturesList, predefinedMouseGesturesList, customKeyboardGesturesList, allGesturesPreview
    
    predefinedKeyboardGesturesList.clear()
    predefinedMouseGesturesList.clear()
    customKeyboardGesturesList.clear()
    reloadKeyMappingData()
    populatePredefinedAndCustomKeyboardGesturesList()
    reloadProfilesData()
    
def reloadKeyMappingData():
    global key_mapping
    key_mapping.clear()
    get_key_mapping()
    
def reloadProfilesData():
    global profileList
    profileList.clear()
    loadProfileList()
    
def reloadGesturesPreview():
    global allGesturesPreview
    allGesturesPreview.clear()
    loadGesturesPreview()

# for now take the profile as int number
def updateCurrentProfile(profile):
    print(f"updateCurrentProfile()")
    global SELECTED_PROFILE, CURRENT_PROFILE_KEYSTROKE_BINDING_PATH, CURRENT_PROFILE_KEYPOINT_PATH, CURRENT_PROFILE_KEYPOINT_LABEL_PATH, isFirstRender, profileList
    if profile == profileList[0]:
        SELECTED_PROFILE = profileList[0]
        isFirstRender = True
        return 
    SELECTED_PROFILE = profile
    CURRENT_PROFILE_KEYSTROKE_BINDING_PATH = KEYSTROKE_BINDING_FILEPATH + KEYSTROKE_BINDING_FILENAME + str(profile) + FILEEXT
    CURRENT_PROFILE_KEYPOINT_PATH = DATASET_PATH + DATASET_FILENAME + str(profile) + FILEEXT
    CURRENT_PROFILE_KEYPOINT_LABEL_PATH = KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + str(profile) + FILEEXT
    isFirstRender = False
    
# call this after updateCurrentProfile()
def checkAndCreateNewCSV():
    print(f"checkAndCreateNewCSV()")
    global CURRENT_PROFILE_KEYSTROKE_BINDING_PATH, CURRENT_PROFILE_KEYPOINT_PATH, CURRENT_PROFILE_KEYPOINT_LABEL_PATH, isFirstRender
    if not isFirstRender:
        if not Path(CURRENT_PROFILE_KEYSTROKE_BINDING_PATH).is_file():
            # Path(KEYSTROKE_BINDING_FILEPATH).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(TEMPLATE_KEYSTROKE_BINDING_FILEPATH, CURRENT_PROFILE_KEYSTROKE_BINDING_PATH)
            print(f"{CURRENT_PROFILE_KEYSTROKE_BINDING_PATH} created")
        else:
            print(f"{CURRENT_PROFILE_KEYSTROKE_BINDING_PATH} already existed.")
            
        if not Path(CURRENT_PROFILE_KEYPOINT_LABEL_PATH).is_file():
            shutil.copy(TEMPLATE_KEYPOINT_LABEL_FILEPATH, CURRENT_PROFILE_KEYPOINT_LABEL_PATH)
            print(f"{CURRENT_PROFILE_KEYPOINT_LABEL_PATH} created")
        else:
            print(f"{CURRENT_PROFILE_KEYPOINT_LABEL_PATH} already existed.")
            
        if not Path(CURRENT_PROFILE_KEYPOINT_PATH).is_file():
            shutil.copy(TEMPLATE_DATASET_FILEPATH, CURRENT_PROFILE_KEYPOINT_PATH)
            print(f"{CURRENT_PROFILE_KEYPOINT_PATH} created")
        else:
            print(f"{CURRENT_PROFILE_KEYPOINT_PATH} already existed.")
            
def show_duplicate_profile_warning():
    messagebox.showwarning("Error", "This is a existing profile name. Please give it another name. Thank you.")
    
def deleteCsvFile():
    print(f"deleteCsvFile()")
    global CURRENT_PROFILE_KEYSTROKE_BINDING_PATH, CURRENT_PROFILE_KEYPOINT_PATH, CURRENT_PROFILE_KEYPOINT_LABEL_PATH
    keystrokeBindingFilePath = Path(CURRENT_PROFILE_KEYSTROKE_BINDING_PATH)
    keypointLabelFilePath = Path(CURRENT_PROFILE_KEYPOINT_LABEL_PATH)
    datasetFilePath = Path(CURRENT_PROFILE_KEYPOINT_PATH)
    
    try:
        os.remove(keystrokeBindingFilePath)
        os.remove(keypointLabelFilePath)
        os.remove(datasetFilePath)
        print(f"Three csv files deleted.")
    except Exception as e:
        print(f"Exception occurs when deleting csv files : {e}")

def deleteProfileFromCSV(profileName):
    print(f"DeleteProfileFromCSV()")
    global profileList, PROFILES_FILEPATH
    
    with open(PROFILES_FILEPATH, "r", newline="", encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row != [profileName]]
    with open(PROFILES_FILEPATH, "w", newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def addNewProfileToCSV(profileName_entry):
    profileName = profileName_entry.get()
    print(f"addNewProfileToCSV()")
    with open(PROFILES_FILEPATH, mode='a', newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([profileName])
    print(f"Added new value to profiles CSV : {profileName}")
        
def setDefaultProfile():
    global isFirstRender
    print(f"setDefaultProfile()")
    isFirstRender = True
    print(f"set default as profile completed")
    
# def reloadAllDataAfterChangingProfile():
    
        
def loadProfileList():
    global profileList
    print(f"loadProfileList()")
    with open(PROFILES_FILEPATH, encoding='utf-8-sig') as f:
        profiles = csv.reader(f)
        profileList = [row[0] for row in profiles]
        print(f"profileList len : {len(profileList)}")
        for row in profileList:
            print(f"profile: {row}")
        
def getGameFont():
    
    pil_font = ImageFont.truetype(font_path, font_size)
    game_font = font.Font(family=pil_font.getname()[0], size=font_size, weight="normal", slant="roman")
    return game_font

def getButtonTopBorder():
    top_border_svg = tksvg.SvgImage(data=button_top_border)
    return top_border_svg

def getButtonBottomBorder():
    bottom_border_svg = tksvg.SvgImage(data=button_bottom_border)
    return bottom_border_svg

def getNextSeqOfKeypointCount():
    print(f"getNextSeqOfKeyPointCount()")
    if isFirstRender:
        with open(KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + FILEEXT,
                encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            labels_list = [row[0] for row in labels]
            return len(labels_list)
        print(f"isFirstRender: {isFirstRender}")
    else:
        with open(CURRENT_PROFILE_KEYPOINT_LABEL_PATH,
                encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            labels_list = [row[0] for row in labels]
            return len(labels_list)
        print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_LABEL_PATH} loaded")
    
def get_key_mapping():
    print("get_key_mapping()")
    if isFirstRender:
        with open(KEYSTROKE_BINDING_FILEPATH + KEYSTROKE_BINDING_FILENAME + FILEEXT,
                encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            for row in labels:
                if len(row) >= 2:
                    gesture,vk_code = row[0], int(row[1])
                    key_mapping[gesture] = vk_code
                    
                    key_name = row[2] if len(row) == 3 else ""
                    key_mapping[gesture] = (vk_code, key_name)
        print(f"isFirstRender: {isFirstRender}")

    else:
        with open(CURRENT_PROFILE_KEYSTROKE_BINDING_PATH,
                encoding='utf-8-sig') as f:
            labels = csv.reader(f)
            for row in labels:
                if len(row) >= 2:
                    gesture,vk_code = row[0], int(row[1])
                    key_mapping[gesture] = vk_code
                    
                    key_name = row[2] if len(row) == 3 else ""
                    key_mapping[gesture] = (vk_code, key_name)
        print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYSTROKE_BINDING_PATH} loaded")


def update_keystroke_binding(gesture, new_VK_Code, key_name):
    print(f"update_keystroke_binding()")
    
    if gesture in key_mapping:
        print(f'Gesture {gesture} in key_mapping')
        key_mapping[gesture] = (new_VK_Code, key_name)
    else:
        print(f'Gesture {gesture} not in key_mapping')
        key_mapping[gesture] = (new_VK_Code, key_name)
        
    if isFirstRender:
        with open(KEYSTROKE_BINDING_FILEPATH + KEYSTROKE_BINDING_FILENAME + FILEEXT, 'w', newline="", encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            for key, (value, name) in key_mapping.items():
                writer.writerow([key, value, name])
        print(f"isFirstRender: {isFirstRender}")

    else:
        with open(CURRENT_PROFILE_KEYSTROKE_BINDING_PATH, 'w', newline="", encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            for key, (value, name) in key_mapping.items():
                writer.writerow([key, value, name])
        print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYSTROKE_BINDING_PATH} loaded")
        

def navigateTo(page):
    if page in frameList:
        frame = frameList[page]
        if page in [MAINMENU_UI, PREDEFINED_HG_KEYBOARD_UI, PREDEFINED_HG_MOUSE_UI, CUSTOM_HG_KEYBOARD_UI]:
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
        
def loadGesturesPreview():
    global isFirstRender, allGesturesPreview
    if isFirstRender:
         with open(KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + FILEEXT,
            encoding='utf-8-sig') as f:
            keypoint_classifier_labels = list(csv.reader(f))
            first_occurrence = {}  # Track first occurrence of each gesture index
            dataset_filepath = DATASET_PATH + DATASET_FILENAME + FILEEXT
            with open(dataset_filepath, encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for row in reader:
                    gesture_index = int(row[0])  # First column is the gesture index
                    coords = row[1:]  # Rest are coordinates

                    if gesture_index not in first_occurrence:
                        first_occurrence[gesture_index] = coords  # Store first occurrence

            # Map previews using labels
            allGesturesPreview = {
                keypoint_classifier_labels[idx][0]: " ".join(first_occurrence[idx]) if idx in first_occurrence else "No Preview"
                for idx in range(len(keypoint_classifier_labels))
            }
            print(f"Previews Extracted: {allGesturesPreview}")
    else:
        with open(CURRENT_PROFILE_KEYPOINT_LABEL_PATH,
            encoding='utf-8-sig') as f:
            keypoint_classifier_labels = list(csv.reader(f))
            first_occurrence = {}  # Track first occurrence of each gesture index
            dataset_filepath = DATASET_PATH + DATASET_FILENAME + FILEEXT
            with open(dataset_filepath, encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                for row in reader:
                    gesture_index = int(row[0])  # First column is the gesture index
                    coords = row[1:]  # Rest are coordinates

                    if gesture_index not in first_occurrence:
                        first_occurrence[gesture_index] = coords  # Store first occurrence

            # Map previews using labels
            allGesturesPreview = {
                keypoint_classifier_labels[idx][0]: " ".join(first_occurrence[idx]) if idx in first_occurrence else "No Preview"
                for idx in range(len(keypoint_classifier_labels))
            }

            print(f"Previews Extracted: {allGesturesPreview}")

def populatePredefinedAndCustomKeyboardGesturesList():
    print(f"populatePredefinedAndCustomKeyboardGesturesList()")
    global predefinedKeyboardGesturesList, customKeyboardGesturesList, predefinedMouseGesturesList, allGesturesPreview
    # allGesturesPreview = {}
    if isFirstRender:
        with open(KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + FILEEXT,
            encoding='utf-8-sig') as f:
            keypoint_classifier_labels = list(csv.reader(f))
            predefinedKeyboardGesturesList = [
                row[0] for row in keypoint_classifier_labels[:4]
            ]
            print(f"populatePredefinedKeyboardGesturesList() result: {len(predefinedKeyboardGesturesList)}")
            predefinedMouseGesturesList = [
                row[0] for row in keypoint_classifier_labels[4:7]
            ]
            print(f"predefinedMouseGesturesList result: {len(predefinedMouseGesturesList)} ")
            if len(keypoint_classifier_labels) > 7:
                customKeyboardGesturesList = [
                    row[0] for row in keypoint_classifier_labels[7:]
                ] 
            print(f"populateCustomKeyboardGesturesList() result: {len(customKeyboardGesturesList)}")
        print(f"isFirstRender: {isFirstRender}")

        first_occurrence = {}  # Track first occurrence of each gesture index
        dataset_filepath = DATASET_PATH + DATASET_FILENAME + FILEEXT
        with open(dataset_filepath, encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                gesture_index = int(row[0])  # First column is the gesture index
                coords = row[1:]  # Rest are coordinates

                if gesture_index not in first_occurrence:
                    first_occurrence[gesture_index] = coords  # Store first occurrence

        # Map previews using labels
        allGesturesPreview = {
            keypoint_classifier_labels[idx][0]: " ".join(first_occurrence[idx]) if idx in first_occurrence else "No Preview"
            for idx in range(len(keypoint_classifier_labels))
        }

        print(f"Previews Extracted: {allGesturesPreview}")
    else:
        with open(CURRENT_PROFILE_KEYPOINT_LABEL_PATH,
            encoding='utf-8-sig') as f:
            keypoint_classifier_labels = list(csv.reader(f))
            predefinedKeyboardGesturesList = [
                row[0] for row in keypoint_classifier_labels[:4]
            ]
            print(f"populatePredefinedKeyboardGesturesList() result: {len(predefinedKeyboardGesturesList)}")
            predefinedMouseGesturesList = [
                row[0] for row in keypoint_classifier_labels[4:7]
            ]
            print(f"predefinedMouseGesturesList result: {len(predefinedMouseGesturesList)} ")
            if len(keypoint_classifier_labels) > 7:
                customKeyboardGesturesList = [
                    row[0] for row in keypoint_classifier_labels[7:]
                ] 
            print(f"populateCustomKeyboardGesturesList() result: {len(customKeyboardGesturesList)}")
        print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_LABEL_PATH} loaded")
        
def produceTrainAndTestDataset():
    X_dataset = np.loadtxt(DATASET_PATH + DATASET_FILENAME + FILEEXT, delimiter=',', dtype='float32', usecols=list(range(1, (21*2)+1)))
    y_dataset = np.loadtxt(DATASET_PATH + DATASET_FILENAME + FILEEXT, delimiter=',', dtype='int32', usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test

def buildModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2,)),  # Input layer for 42 coordinates
        tf.keras.layers.BatchNormalization(),  # Normalize input for better performance
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Wider layer
        tf.keras.layers.LeakyReLU(alpha=0.1),  # Leaky ReLU activation with a small slope
        tf.keras.layers.Dropout(0.3),  # Moderate dropout for regularization
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Another hidden layer
        tf.keras.layers.LeakyReLU(alpha=0.1),  # Leaky ReLU activation
        tf.keras.layers.Dropout(0.2),  # Lower dropout for a smaller layer
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Output layer for classification
    ])
    return model

def compileModel(model):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),  # Adam with weight decay
        loss='sparse_categorical_crossentropy',  # Works for integer-encoded labels
        metrics=['accuracy']  # Track accuracy
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
            model = buildModel()
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
    print(f'show_loading_popup start')
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
    print(f'show_loading_popup started')

    return loading_window, spinner

def hide_loading_popup(loading_window):
    # Close the loading window
    loading_window.destroy()

def addHandGestureToCSV(newHG):
    print("addHandGestureToCSV()")
    if isFirstRender:
        with open(KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + FILEEXT, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([newHG])
            print(f"Added new value to keypoint CSV: {newHG}")
        print(f"isFirstRender: {isFirstRender}")
    else:
        with open(CURRENT_PROFILE_KEYPOINT_LABEL_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([newHG])
            print(f"Added new value to keypoint CSV: {newHG}")
        print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_LABEL_PATH} loaded")

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
            
    response = messagebox.askyesno("Warning", "Dear player, are you sure you want to delete the gesture "+str(customHGName)+"?")
    if response:
        loading_window, spinner = show_loading_popup(frame, DELETING_LABEL)
        threading.Thread(target=task, daemon=True).start()

    # # Show the loading spinner
    # loading_window, spinner = show_loading_popup(frame, DELETING_LABEL)

    # # Run the task in a separate thread
    # threading.Thread(target=task, daemon=True).start()
    
def deleteHandGestureFromCSV(gesture):
    print(f"deleteHandGestureFromCSV()")
    if isFirstRender:
        with open(KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + FILEEXT, mode='r', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
        print(f"isFirstRender: {isFirstRender}")
    else:
        with open(CURRENT_PROFILE_KEYPOINT_LABEL_PATH, mode='r', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
        print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_LABEL_PATH} loaded")
        
    # print(f"rows: {rows}")
    index = 0
    for rowIndex, row in enumerate(rows):
        if gesture in row:
            index = rowIndex
    updated_rows = [row for row in rows if row and row[0] != gesture]
    
    if isFirstRender:
        with open(KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + FILEEXT, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)
        print(f"isFirstRender: {isFirstRender}")
    else:
        with open(CURRENT_PROFILE_KEYPOINT_LABEL_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)
        print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_LABEL_PATH} loaded")

    
    print(f"Deleted value from keypoint csv: {gesture}")
    return index

def updateKeypointCSV(index):
    print(f"updateKeypointCSV()")
    dataframe = pd.read_csv(DATASET_PATH + DATASET_FILENAME + FILEEXT, header=None, engine="python", on_bad_lines="skip")
    updatedDataframe = dataframe[dataframe[0] != index].reset_index(drop=True)
    updatedDataframe[0] = updatedDataframe[0].where(updatedDataframe[0] < index, updatedDataframe[0]-1)
    updatedDataframe.to_csv(DATASET_PATH + DATASET_FILENAME + FILEEXT, index=False, header=False)

class CustomHandGestureObject:
    def __init__(self, name):
        self.name = name
        
class Root(tk.Tk):
    def __init__(self):
        super().__init__()
        loadProfileList()
        populatePredefinedAndCustomKeyboardGesturesList()
        get_key_mapping()
        for key, (value, name) in key_mapping.items():
            print(f"Gesture: {key}, VK_Code: {str(value)}, key_name: {name}")
        
        self.title("Playing Games with Bare Hands")
        self.geometry("800x800")
        self.configure(bg="#f0f0f0")
        
        # scrollableContainer = ScrollableFrame(self)
        # scrollableContainer.pack(fill="both", expand=True)
        
        # mainFrame = Frame(scrollableContainer.scrollable_frame)
        # mainFrame.pack(fill="both", expand=True)
        
        mainFrame = Frame(self)
        mainFrame.pack(fill="both", expand=True)
        
        mainFrame.grid_rowconfigure(0, weight=1)
        mainFrame.grid_columnconfigure(0, weight=1)
        
        # loadProfileList()
        
        # if not isFirstRender:
        #     updateCurrentProfile(SELECTED_PROFILE)
        #     checkAndCreateNewCSV()
            
        # populatePredefinedAndCustomKeyboardGesturesList()
        # get_key_mapping()
        # for key, (value, name) in key_mapping.items():
        #     print(f"Gesture: {key}, VK_Code: {str(value)}, key_name: {name}")
        # self.load_game_font()
        # self.frames = {}
        
        for page in (MainMenuUI, PredefinedHandGesturesUI, CustomHandGesturesUI, PredefinedHandGesturesKeyboardUI, PredefinedHandGesturesMouseUI, CustomHandGesturesKeyboardUI, NewCustomHandGestureForKeyboard, NewProfileComponent, OptionUI):
            print(f"Initializing frame for {page.getIdentity()}")
            frame = page(mainFrame, self)
            # if page.getIdentity() in [PREDEFINED_HG_KEYBOARD_UI, PREDEFINED_HG_MOUSE_UI, CUSTOM_HG_KEYBOARD_UI]:
            #     page.populatePageElements(frame)
            frameList[page.getIdentity()] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
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

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.window = self.canvas.create_window((0,0), window=self.scrollable_frame, anchor="nw")
        self.bind("<Configure>", self.update_canvas_width)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def update_canvas_width(self):
        self.canvas.itemconfig(self.window, width=self.canvas.winfo_width())

class ProfileSelection(ttk.Frame):
    global profileList, SELECTED_PROFILE
    def __init__(self, parent):
        super().__init__(parent)
        
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.profiles_option = StringVar()
        self.profiles_option.set(SELECTED_PROFILE)
        self.profiles_option.trace_add("write", self.onProfileSelected)
        
        # Create OptionMenu
        self.profilesMenu = OptionMenu(self, self.profiles_option, *profileList)
        self.profilesMenu.config(
            bg="black",
            fg="#FFF",
            # activebackground="#FFF",
            activeforeground="black",
            font=("Venite Adoremus", 16),
            border=0,
            highlightthickness=1,
            highlightbackground="black",
            pady=20,
            indicatoron=0,
            width=25  # Matching width with buttons
        )
        self.profilesMenu.grid(row=0, column=0, padx=10, pady=10)

        # Add Button
        self.profilesAddBtn = buildButton(self, "Add", navigateTo, NEW_PROFILE_UI)
        self.profilesAddBtn.config(width=10)  # Matching width
        self.profilesAddBtn.grid(row=0, column=1, padx=10, pady=10)

        # Delete Button
        self.profilesDeleteBtn = buildButton(self, "Delete", self.delete_profile, None)
        self.profilesDeleteBtn.config(width=10)  # Matching width
        self.profilesDeleteBtn.grid(row=0, column=2, padx=10, pady=10)
        
    def populatePageElements(self):
        for widget in self.winfo_children():
            widget.destroy()
            
        self.profiles_option = StringVar()
        self.profiles_option.set(SELECTED_PROFILE)
        self.profiles_option.trace_add("write", self.onProfileSelected)

        # Create OptionMenu
        self.profilesMenu = OptionMenu(self, self.profiles_option, *profileList)
        self.profilesMenu.config(
            bg="black",
            fg="#FFF",
            # activebackground="#FFF",
            activeforeground="black",
            font=("Venite Adoremus", 16),
            border=0,
            highlightthickness=1,
            highlightbackground="black",
            pady=20,
            indicatoron=0,
            width=25  # Matching width with buttons
        )
        self.profilesMenu.grid(row=0, column=0, padx=10, pady=10)

        # Add Button
        self.profilesAddBtn = buildButton(self, "Add", navigateTo, NEW_PROFILE_UI)
        self.profilesAddBtn.config(width=10)  # Matching width
        self.profilesAddBtn.grid(row=0, column=1, padx=10, pady=10)

        # Delete Button
        self.profilesDeleteBtn = buildButton(self, "Delete", self.delete_profile, None)
        self.profilesDeleteBtn.config(width=10)  # Matching width
        self.profilesDeleteBtn.grid(row=0, column=2, padx=10, pady=10)
        
    def onProfileSelected(self, *args):
        selectedProfile = self.profiles_option.get()
        print(f"Profile {selectedProfile} selected")
        updateCurrentProfile(selectedProfile)
        checkAndCreateNewCSV()
        reloadProfileDependantData()
            
    def delete_profile(self):
        if len(profileList) == 1:
            messagebox.showwarning("Warning","At least one profile must exist and cannot be deleted.")
        else:
            response = messagebox.askyesno("Warning", "Dear player, are you sure you want to delete the profile "+ str(SELECTED_PROFILE)+"?")
            if response:
                profileList.remove(SELECTED_PROFILE)
                deleteCsvFile()
                deleteProfileFromCSV(SELECTED_PROFILE)
                print(f"profile {SELECTED_PROFILE} deleted")
                updateCurrentProfile(profileList[0])
                self.profiles_option.set(SELECTED_PROFILE)
                self.populatePageElements()

class MainMenuUI(ttk.Frame):
    global isTurboChecked, profileList, SELECTED_PROFILE
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame)
        self.root = root
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=mainMenuTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        optionBtn =buildButton(self, "Options", navigateTo, OPTION_UI)
        frameButton(self, optionBtn)

        # customHandGesturesBtn = buildButton(self, "Custom Hand Gestures", navigateTo, CUSTOM_HG_UI)
        # frameButton(self, customHandGesturesBtn)
        
        testingBtn = Button(
            self,
            text="Test Hand Gestures",
            command= lambda: initiateWebCam(self, 3),
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
        frameButton(self, testingBtn)
        
        startGameBtn = Button(
            self,
            text="Start Game",
            command= lambda: initiateWebCam(self, 1),
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
        frameButton(self, startGameBtn)
        
        # label = Label(self, text="Turbo:", font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", anchor="w", justify="center")  # Align the text to the left
        # label.grid(padx=20, pady=20)

        # self.turboChecked = tk.BooleanVar()
        # self.turboChecked.trace_add("write", self.onTurboChecked)
        # checkbox = ttk.Checkbutton(self, variable=self.turboChecked)
        # # isTurboChecked = self.checkbox_var.get()
        # print(f"checkbox_var: {self.turboChecked.get()}")
        # checkbox.grid(padx=20, pady=20)
        
        # self.profiles_option = StringVar()
        
        # print(f"profilelist size: {len(profileList)}")
        
        # Set default profile
        # self.profiles_option.set(profileList[0])
        # self.profiles_option.trace_add("write", self.onProfileSelected)
        
        # profilesMenu = OptionMenu(self, self.profiles_option, *profileList)
        # profilesMenu.config(
        #     bg="black",
        #     fg="#FFF",
        #     activebackground="#FFF",
        #     activeforeground="black",
        #     font=("Venite Adoremus", 16),
        #     border=0,
        #     highlightthickness=1,
        #     highlightbackground="black",
        #     pady=20,
        #     indicatoron=0
        # ) 
        # profilesMenu.grid(row=10, column=0, padx=20, pady=20, sticky="w")
        
        # profilesAddBtn = buildButton(self, "Add", navigateTo, NEW_PROFILE_UI)
        # profilesAddBtn.grid(row=10, column=1, pady=20, padx=20)
        
        # profilesDeleteBtn = buildButton(self, "Delete", self.delete_profile, None)
        # profilesDeleteBtn.grid(row=10, column=2, pady=20, padx=20)
        
        # profileButton = buildButton(self, "Select Profile", self.show, None)
        # profileButton.grid(padx=20, pady=20)
        
        self.profileFrame = ProfileSelection(self)
        self.profileFrame.grid(row=10, column=0, columnspan=3, pady=5, padx=10, sticky= "ew")
        
    def populatePageElements(self):
        print(f"MainMenuUI populatePageElements()")
        
        for widget in self.winfo_children():
            widget.destroy()
            
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=mainMenuTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        optionBtn =buildButton(self, "Options", navigateTo, OPTION_UI)
        frameButton(self, optionBtn)

        # customHandGesturesBtn = buildButton(self, "Custom Hand Gestures", navigateTo, CUSTOM_HG_UI)
        # frameButton(self, customHandGesturesBtn)
        
        testingBtn = Button(
            self,
            text="Test Hand Gestures",
            command= lambda: initiateWebCam(self, 3),
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
        frameButton(self, testingBtn)
        
        startGameBtn = Button(
            self,
            text="Start Game",
            command= lambda: initiateWebCam(self, 1),
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
        frameButton(self, startGameBtn)
        
        self.profileFrame = ProfileSelection(self)
        self.profileFrame.grid(row=10, column=0, columnspan=3, pady=5, padx=10, sticky= "ew")
        
    # def onProfileSelected(self, *args):
    #     selectedProfile = self.profiles_option.get()
    #     print(f"Profile {selectedProfile} selected")
    #     updateCurrentProfile(selectedProfile)
    #     checkAndCreateNewCSV()
    #     reloadProfileDependantData()
            
    # def delete_profile(self):
    #     if len(profileList) == 1:
    #         messagebox.showwarning("Warning","At least one profile must exist and cannot be deleted.")
    #     else:
    #         response = messagebox.askyesno("Warning", "Dear player, are you sure you want to delete the profile "+ str(SELECTED_PROFILE)+"?")
    #         if response:
    #             profileList.remove(SELECTED_PROFILE)
    #             deleteCsvFile()
    #             deleteProfileFromCSV(SELECTED_PROFILE)
    #             print(f"profile {SELECTED_PROFILE} deleted")
    #             updateCurrentProfile(profileList[0])
    #             self.profiles_option.set(SELECTED_PROFILE)
    #             self.profileFrame.populatePageElements()
        
    def onTurboChecked(self, *args):
        global isTurboChecked
        isTurboChecked = self.turboChecked.get()
        print(f"isTurboChecked: {isTurboChecked}")
        
    def show(self):
        self.profileLabel.config(text=self.profiles_option.get())
        
    def getIdentity():
        return MAINMENU_UI

class OptionUI(ttk.Frame):
    global isTurboChecked
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame)
        self.root = root
        
        loadWallpaper(self)
        loadProductName(self)
        
        title = Label(self, text=optionTitle, font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", justify="center")
        title.grid(pady=10)
        
        predefinedHandGesturesBtn =buildButton(self, "Predefined Hand Gestures", navigateTo, PREDEFINED_HG_UI)
        frameButton(self, predefinedHandGesturesBtn)

        customHandGesturesBtn = buildButton(self, "Custom Hand Gestures", navigateTo, CUSTOM_HG_UI)
        frameButton(self, customHandGesturesBtn)
        
        label = Label(self, text="Turbo:", font=("Venite Adoremus", 25, 'bold'), fg="#FFF", bg="black", anchor="w", justify="center")  # Align the text to the left
        label.grid(padx=20, pady=20)

        self.turboChecked = tk.BooleanVar()
        self.turboChecked.trace_add("write", self.onTurboChecked)
        checkbox = ttk.Checkbutton(self, variable=self.turboChecked)
        # isTurboChecked = self.checkbox_var.get()
        print(f"checkbox_var: {self.turboChecked.get()}")
        checkbox.grid(padx=20, pady=20)
        
        doneBtn = buildDoneButton(self, "Done", navigateTo, MAINMENU_UI)
        frameButton(self, doneBtn)
        
    def onTurboChecked(self, *args):
        global isTurboChecked
        isTurboChecked = self.turboChecked.get()
        print(f"isTurboChecked: {isTurboChecked}")

    def getIdentity():
        return OPTION_UI
        
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
        doneBtn = buildDoneButton(self, "Done", navigateTo, OPTION_UI)
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
        doneBtn = buildDoneButton(self, "Done", navigateTo, OPTION_UI)
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
        print(f"PredefinedHandGesturesKeyboardUI predefinedKeyboardGesturesList: {len(predefinedKeyboardGesturesList)} {predefinedKeyboardGesturesList}")
        
        for widget in self.winfo_children():
            widget.destroy()
            
        loadWallpaper(self)
        loadProductName(self)
        reloadGesturesPreview()
        
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
            if idx < len(predefinedKeyboardGesturesList):

                componentRow = idx // MAX_COLUMN
                componentColumn = idx % MAX_COLUMN
                
                hgComponent = PredefinedHandGestureComponent.HandGestureComponent(self, label_text=gesture, button_command=self.enable_recording, key_mapping = key_mapping, isMouse=False, coords=allGesturesPreview[predefinedKeyboardGesturesList[idx]])
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
        if self.checkIfKeystrokeBound(vk_code):
            key_mapping[gesture] = (vk_code, key_name)
            label.config(text=f"{gesture}: {key_name}", font=("Venite Adoremus", 10, 'bold'), foreground="green")
            self.root.unbind("<Key>")
            update_keystroke_binding(gesture, vk_code, key_name)
            keystroke_binding[gesture] = key_name
        key_mapping[gesture] = (vk_code, key_name)
        self.root.unbind("<Key>")

    # Define a function to enable key recording
    def enable_recording(self, gesture, label):
        label.config(text=f"{gesture}: Press any key...", foreground="orange")
        self.root.bind("<Key>", lambda event: self.record_key(event, gesture, label))
    
    def show_conflict_warning(self):
        response = messagebox.askyesno("Warning", "Please note that this keystroke was bound to other hand gesture. Do you still want to bind?")
        if response:
            print("Player bind existing keystroke.")
            return True
        else:
            print("Player reject to bind existing keystroke.")
            return False
        
    # Return True to bind keystroke
    def checkIfKeystrokeBound(self, vk_code):
        vk_codes = [value[0] for value in key_mapping.values()]
        if vk_code in vk_codes:
            print("Found conflicted keystroke")
            return self.show_conflict_warning()
        else:
            print("No conflicted keystroke")
            return True

        
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
    global predefinedMouseGesturesList
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
        print(f"PredefinedHandGesturesMouseUI predefinedMouseGesturesList: {len(predefinedMouseGesturesList)} {predefinedMouseGesturesList}")
        
        for widget in self.winfo_children():
            widget.destroy()
            
        loadWallpaper(self)
        loadProductName(self)
        reloadGesturesPreview()
        
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
            
            # in key_mapping, only has 7 default gestures and mapping, without the new added custom gesture
            hgComponent = PredefinedHandGestureComponent.HandGestureComponent(self, label_text=gesture, button_command=self.enable_recording, key_mapping=key_mapping, isMouse=True, coords=allGesturesPreview[predefinedMouseGesturesList[idx]])
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
        if self.checkIfKeystrokeBound(vk_code):
            key_mapping[gesture] = (vk_code, key_name)
            label.config(text=f"{gesture}: {key_name}", font=("Venite Adoremus", 10, 'bold'), foreground="green")
            self.root.unbind("<Key>")
            update_keystroke_binding(gesture, vk_code, key_name)
            keystroke_binding[gesture] = key_name
        key_mapping[gesture] = (vk_code, key_name)
        self.root.unbind("<Key>")

    # Define a function to enable key recording
    def enable_recording(self, gesture, label):
        label.config(text=f"{gesture}: Press any key...", foreground="orange")
        self.root.bind("<Key>", lambda event: self.record_key(event, gesture, label))
        
    def show_conflict_warning(self):
        response = messagebox.askyesno("Warning", "Please note that this keystroke was bound to other hand gesture. Do you still want to bind?")
        if response:
            print("Player bind existing keystroke.")
            return True
        else:
            print("Player reject to bind existing keystroke.")
            return False
    
    # Return True to bind keystroke
    def checkIfKeystrokeBound(self, vk_code):
        if vk_code in key_mapping.values():
            print("Found conflicted keystroke")
            return self.show_conflict_warning()
        else:
            print("No conflicted keystroke")
            return True

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
    global customKeyboardGesturesList
    
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
        self.populatePageElements()
        
    def populatePageElements(self):
        print(f"CustomHandGesturesKeyboardUI populatePageElements()")
        print(f"CustomHandGesturesKeyboardUI customKeyboardGesturesList: {len(customKeyboardGesturesList)} {customKeyboardGesturesList}")
        
        for widget in self.winfo_children():
            widget.destroy()
            
        loadWallpaper(self)
        loadProductName(self)
        reloadGesturesPreview()
        
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
            
        total_rows = (len(customKeyboardGesturesList) + MAX_COLUMN - 1) // MAX_COLUMN
        for row in range(total_rows + 2):
            self.grid_rowconfigure(row, weight=1, uniform="row")
        
        # Add labels and buttons for each gesture
        if customKeyboardGesturesList:
            for idx, customHGName in enumerate(customKeyboardGesturesList):
                
                componentRow = idx // MAX_COLUMN
                componentColumn = idx % MAX_COLUMN
                
                hgComponent = CustomHandGestureComponent.HandGestureComponent(self, label_text=customHGName, button_command=self.enable_recording, delete_button_command=deleteCustomGesture, key_mapping=key_mapping, coords=allGesturesPreview[customKeyboardGesturesList[idx]])
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
        if self.checkIfKeystrokeBound(vk_code):
            key_mapping[gesture] = (vk_code, key_name)
            label.config(text=f"{gesture}: {key_name}", font=("Venite Adoremus", 10, 'bold'), foreground="green")
            self.root.unbind("<Key>")
            update_keystroke_binding(gesture, vk_code, key_name)
            keystroke_binding[gesture] = key_name
        key_mapping[gesture] = (vk_code, key_name)
        self.root.unbind("<Key>")

    # Define a function to enable key recording
    def enable_recording(self, gesture, label):
        label.config(text=f"{gesture}: Press any key...", foreground="orange")
        self.root.bind("<Key>", lambda event: self.record_key(event, gesture, label))
        
    def show_conflict_warning(self):
        response = messagebox.askyesno("Warning", "Please note that this keystroke was bound to other hand gesture. Do you still want to bind?")
        if response:
            print("Player bind existing keystroke.")
            return True
        else:
            print("Player reject to bind existing keystroke.")
            return False
    
    # Return True to bind keystroke
    def checkIfKeystrokeBound(self, vk_code):
        if vk_code in key_mapping.values():
            print("Found conflicted keystroke")
            return self.show_conflict_warning()
        else:
            print("No conflicted keystroke")
            return True
        
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
        self.grid_columnconfigure(1, weight=1)
        instructions = """1. Enter Custom Hand Gesture name.
2. Click on "Click to Record" to start recording the hand gesture.
3. Click on "Train" to train and save the custom hand gesture."""
        
        loadWallpaper(self)
        loadProductName(self)
        
        instruction_label = Label(self, text=instructions, font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black", justify="center")
        instruction_label.grid(row=1, column=0, columnspan=2, padx=(0, 10), pady=5, sticky="ew")
        newHG_label = Label(self, text='Hand Gesture Name', font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black", justify="center")
        newHG_label.grid(row=2, column=0, padx=(0, 10), pady=5, sticky="e")
        newHG_entry = Entry(self, textvariable=self.newHGName, font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black", justify="center")
        newHG_entry.grid(row=2, column=1, pady=5, sticky="w")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        
        keypointCount = getNextSeqOfKeypointCount()
        print(f"Total count in keypoints csv: {keypointCount}")
                
        add_gesture_button = Button(
            self,
            text="Click to record",
            command= lambda: initiateWebCam(self, 2),
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
        add_gesture_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # done_button = buildDoneButton(self, "Done", addNewCustomGesture, newHG_entry)
        done_button = Button(
            self,
            text="Train",
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
        done_button.grid(row=5, column=0, columnspan=2, pady=20)
        
        back_button = buildButton(self, "Back", navigateTo, CUSTOM_HG_KEYBOARD_UI)
        back_button.grid(row=6, column=0, columnspan=2, pady=20)
    
    def getIdentity():
        return NEW_CUSTOM_HG_UI
    
class NewProfileComponent(ttk.Frame):
    global profileList
    
    def __init__(self, mainFrame, root):
        super().__init__(mainFrame, padding=20)
        self.root = root
        self.newProfileName = ''
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)
        
        loadWallpaper(self)
        loadProductName(self)
        
        newProfile_label = Label(self, text='Profile\'s name', font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black", justify="center")
        newProfile_label.grid(row=1, column=0, padx=(0, 10), pady=5, sticky="e")
        newProfile_entry = Entry(self, textvariable=self.newProfileName, font=("Venite Adoremus", 15, 'bold'), fg="#FFF", bg="black", justify="center")
        newProfile_entry.grid(row=1, column=1, pady=5, sticky="w")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 12), padding=5)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
        style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
        style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))
        
        print(f"Total count in profiles: {len(profileList)}")
        
        # done_button = buildDoneButton(self, "Done", addNewCustomGesture, newHG_entry)
        done_button = Button(
            self,
            text="Done",
            command= lambda: self.add_profiles(newProfile_entry),
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
        done_button.grid(row=2, column=0, columnspan=2, pady=20)
        
        back_button = buildButton(self, "Back", navigateTo, MAINMENU_UI)
        back_button.grid(row=3, column=0, columnspan=2, pady=20)
        
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
        else:
            messagebox.showwarning("Warning", "Please give at least a alphabet to as your new custom hand gesture name")
        
    def isNewProfileExists(self, newProfile_entry):
        enteredProfileName = newProfile_entry.get()
        print(f"isNewProfileExists(), argument: {enteredProfileName}")
        for name in profileList:
            if name == enteredProfileName:
                print(f"{enteredProfileName} profile already exists")
                return True
        print(f"{enteredProfileName} does not exist")
        return False
        
    def add_profiles(self, newProfile_entry):
        print(f"add_profiles()")
        if newProfile_entry.get() != "":
            isProfileExist = self.isNewProfileExists(newProfile_entry)
            if not isProfileExist:
                print(f"profileList count before appending : {len(profileList)}")
                profileList.append(newProfile_entry.get())
                print(f"profileList count after appending : {len(profileList)}")
                addNewProfileToCSV(newProfile_entry)
                reloadProfilesData()
                newProfile_entry.delete(0, "end")
                # updateCurrentProfile(newProfile_entry.get())
                # checkAndCreateNewCSV()
                navigateTo(MAINMENU_UI)
            else:
                show_duplicate_profile_warning()
        else:
            messagebox.showwarning("Warning", "You need to add at least one alphabet as your profile name.")
        
    def getIdentity():
        return NEW_PROFILE_UI
        
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

# key_mapping = {
#     "Up": 0x26,       # VK_UP (Arrow Up)
#     "Down": 0x28,     # VK_DOWN (Arrow Down)
#     "Left": 0x25,     # VK_LEFT (Arrow Left)
#     "Right": 0x27,    # VK_RIGHT (Arrow Right)
#     # Add mouse clicks to key_mapping
#     "LeftClick": 0x01,    # Left mouse button
#     "RightClick": 0x02,   # Right mouse button
# }

THROTTLE_TIME = 0.1
last_key_press_time = 0

def hand_cursor_control(cursor_x, cursor_y):
    # Get current screen resolution
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    
    # Ensure coordinates are within screen bounds
    cursor_x = max(0, min(cursor_x, screen_width))
    cursor_y = max(0, min(cursor_y, screen_height))
    
    # Convert to absolute coordinates (required by SetCursorPos)
    absolute_x = int(65535 * cursor_x / screen_width)
    absolute_y = int(65535 * cursor_y / screen_height)
    
    # Move cursor using win32api
    win32api.mouse_event(win32con.MOUSEEVENTF_ABSOLUTE | win32con.MOUSEEVENTF_MOVE, 
                        absolute_x, absolute_y, 0, 0)

async def press_key(key_name, is_turbo=False):
    global last_key_press_time

    # Ensure the key_name is valid
    if key_name not in key_mapping:
        return

    # Get the hex key code for the provided key name
    hex_key_code = key_mapping[key_name][0]

    current_time = time.time()
    
    # Handle mouse clicks differently from keyboard events
    if key_name in ["LeftClick", "RightClick", "MiddleClick"]:
        if is_turbo:
            # For turbo mode, hold down the mouse button
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN if key_name == "LeftClick" else
                               win32con.MOUSEEVENTF_RIGHTDOWN if key_name == "RightClick" else
                               win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
        else:
            # For normal mode, click and release if throttle time has passed
            if current_time - last_key_press_time >= THROTTLE_TIME:
                # Press mouse button
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN if key_name == "LeftClick" else
                                   win32con.MOUSEEVENTF_RIGHTDOWN if key_name == "RightClick" else
                                   win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                await asyncio.sleep(0.05)  # Short delay
                # Release mouse button
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP if key_name == "LeftClick" else
                                   win32con.MOUSEEVENTF_RIGHTUP if key_name == "RightClick" else
                                   win32con.MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                last_key_press_time = current_time
    else:
        # Handle keyboard events as before
        if is_turbo:
            win32api.keybd_event(hex_key_code, 0, 0, 0)
        else:
            if current_time - last_key_press_time >= THROTTLE_TIME:
                win32api.keybd_event(hex_key_code, 0, 0, 0)
                await asyncio.sleep(0.05)
                win32api.keybd_event(hex_key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                last_key_press_time = current_time
                # print(f"Key {key_name} pressed (Throttled mode).")

# WebCam processing function
def initiateWebCam(frame, isGameStart):
    print(f"initiateWebCam()")
    async def asyncTask():
        try:
            print(f'Initializing Web Camera start')
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
            cap.set(cv.CAP_PROP_FPS, 60) 
            print("WebCam Initiated")

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=use_static_image_mode,
                max_num_hands=2,  # Reduce to 1 hand for faster processing
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            print("Mediapipe Initiated")
            keypoint_classifier = KeyPointClassifier()
            # point_history_classifier = PointHistoryClassifier()

            if isFirstRender:
                with open(KEYPOINT_LABEL_FILEPATH + KEYPOINT_LABEL_FILENAME + FILEEXT, encoding='utf-8-sig') as f:
                    keypoint_classifier_labels = csv.reader(f)
                    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
                print(f"isFirstRender: {isFirstRender}")

            else:
                with open(CURRENT_PROFILE_KEYPOINT_LABEL_PATH, encoding='utf-8-sig') as f:
                    keypoint_classifier_labels = csv.reader(f)
                    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
                print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_LABEL_PATH} loaded")


            # with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            #     point_history_classifier_labels = csv.reader(f)
            #     point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]
            print("Gesture Model Initiated")
            cvFpsCalc = CvFpsCalc(buffer_len=5)
            print("FPS Calulator Initiated")
            # history_length = 16
            # point_history = deque(maxlen=history_length)
            # finger_gesture_history = deque(maxlen=history_length)
            mode = 0
            while True:
                fps = cvFpsCalc.get()

                key = cv.waitKey(1)
                if key == 27:  # ESC
                    break
                number, mode = select_mode(key, mode)

                ret, image = cap.read()
                if not ret:
                    break
                image = cv.flip(image, 1)
                debug_image = image.copy()

                # Resize image for faster processing
                image = cv.resize(image, (160, 120))

                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        bottom_palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        cursor_x = int(bottom_palm.x * 1920) #added multiplier
                        cursor_y = int(bottom_palm.y * 1080) #added multiplier

                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        # pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                        logging_csv(number, mode, pre_processed_landmark_list)

                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        # if hand_sign_id == 2:
                        #     point_history.append(landmark_list[8])
                        # else:
                        #     point_history.append([0, 0])

                        # finger_gesture_id = 0
                        # point_history_len = len(pre_processed_point_history_list)
                        # if point_history_len == (history_length * 2):
                        #     finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                        # finger_gesture_history.append(finger_gesture_id)
                        # most_common_fg_id = Counter(finger_gesture_history).most_common()

                        gesture_text = keypoint_classifier_labels[hand_sign_id]

                        # debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(
                            debug_image,
                            brect,
                            handedness, # left or right
                            gesture_text, # class inferred by CNN
                            # point_history_classifier_labels[most_common_fg_id[0][0]],
                        )

                        if isGameStart == 1:
                            if gesture_text == "MouseMove" or gesture_text == "LeftClick" or gesture_text == "RightClick":
                                hand_cursor_control(cursor_x, cursor_y)
                            if isGameStart == 1:
                                if isTurboChecked:
                                    await press_key(gesture_text, is_turbo=True)
                                else:
                                    await press_key(gesture_text, is_turbo=False)
                            if isTurboChecked:
                                await press_key(gesture_text, is_turbo=True)  # Non-blocking turbo mode keypress
                            else:
                                await press_key(gesture_text, is_turbo=False)  # Non-blocking throttled mode keypress
                # else:
                #     point_history.append([0, 0])

                # debug_image = draw_point_history(debug_image, point_history)
                debug_image = draw_info(isGameStart,debug_image, fps, mode, number)

                cv.imshow('HGR To Play Games', debug_image)
                frame.after(0, hide_loading_popup, loading_window)

                await asyncio.sleep(0)  # Yield control to other async tasks

            cap.release()
            cv.destroyAllWindows()
            print(f'Initializing Web Cam end')
        finally:
            print("Initiate Web Cam and display loading dialog successfully.")
      
    def run_async_task():
        newLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(newLoop)
        newLoop.run_until_complete(asyncTask())
        newLoop.close()      
        
    loading_window, spinner = show_loading_popup(frame, INITIATE_WEBCAM_LABEL)
    threading.Thread(target=run_async_task, daemon=True).start()

def get_args():
    parser = argparse.ArgumentParser()

    def find_first_camera():
        for device_id in range(10):  # Scan up to 10 possible devices
            cap = cv.VideoCapture(device_id)
            if cap.isOpened():
                cap.release()
                print("Device ID: ", device_id)
                return device_id  # Return the first available camera
        return 0  # Default to 0 if no cameras are found

    # Auto-detect first available device
    best_device = find_first_camera()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=best_device)
    args = parser.parse_args()
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=690)

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
    # Create the tkiner UI root object declared above
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
    if key == 32:   # space bar
        mode = 3
        number = getNextSeqOfKeypointCount()
        # print(f"Spacebar pressed. Number of gesture in keypoint.csv: {number}")
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

# def pre_process_point_history(image, point_history):
#     image_width, image_height = image.shape[1], image.shape[0]

#     temp_point_history = copy.deepcopy(point_history)

#     base_x, base_y = 0, 0
#     for index, point in enumerate(temp_point_history):
#         if index == 0:
#             base_x, base_y = point[0], point[1]

#         temp_point_history[index][0] = (temp_point_history[index][0] -
#                                         base_x) / image_width
#         temp_point_history[index][1] = (temp_point_history[index][1] - 
#                                         base_y) / image_height

#     temp_point_history = list(
#         itertools.chain.from_iterable(temp_point_history))

#     return temp_point_history

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        if isFirstRender:
            csv_path = DATASET_PATH + DATASET_FILENAME + FILEEXT
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
            print(f"isFirstRender: {isFirstRender} for logging_csv()")
        else:
            csv_path = CURRENT_PROFILE_KEYPOINT_PATH
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
            print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_PATH} loaded for logging_csv()")
    # if mode == 2 and (0 <= number <= 9):
    #     csv_path = 'model/point_history_classifier/point_history.csv'
    #     with open(csv_path, 'a', newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([number, *point_history_list])
    if mode == 3:
        if isFirstRender:
            csv_path = DATASET_PATH + DATASET_FILENAME + FILEEXT
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([getNextSeqOfKeypointCount(), *landmark_list])
                print(f"Write into keypoint.csv for row {getNextSeqOfKeypointCount()+1}")
            print(f"isFirstRender: {isFirstRender} for logging_csv()")
        else:
            csv_path = CURRENT_PROFILE_KEYPOINT_PATH
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([getNextSeqOfKeypointCount(), *landmark_list])
                print(f"Write into keypoint.csv for row {getNextSeqOfKeypointCount()+1}")
            print(f"isFirstRender: {isFirstRender}, {CURRENT_PROFILE_KEYPOINT_PATH} loaded for logging_csv()")

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

    # for index, landmark in enumerate(landmark_point):
    #     if index == 0:  # wrist
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 1:  # wrist
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 2:  # thumb base
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 3:  # thumb joint 1
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 255),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 4:  # thumb tip
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 255),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    #     if index == 5:  # index base
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 6:  # index joint 2
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 7:  # index joint 1
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 8:  # index tip
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (0, 255, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    #     if index == 9:  # middle base
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 10:  # middle joint 2
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 11:  # middle joint 1
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 12:  # middle tip
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (255, 0, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    #     if index == 13:  # ring base
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 100),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 14:  # ring joint 2
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 100),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 15:  # ring joint 1
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 100),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 16:  # ring tip
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (255, 0, 100),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    #     if index == 17:  # pinky base
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (69, 100, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 18:  # pinky joint 2
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (69, 100, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 19:  # pinky joint 1
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (69, 100, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    #     if index == 20:  # pinky tip
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (69, 100, 0),
    #                   -1)
    #         cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
    return image

# def draw_bounding_rect(use_brect, image, brect):
#     if use_brect:
#         cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
#                      (0, 0, 0), 1)

#     return image

def draw_info_text(image, brect, handedness, hand_sign_text): # params can add back finger_gesture_text if history is needed
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image

# def draw_point_history(image, point_history):
#     for index, point in enumerate(point_history):
#         if point[0] != 0 and point[1] != 0:
#             cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
#                       (152, 251, 152), 2)

#     return image

def draw_info(isGameStart, image, fps, mode, number):
    if isGameStart == 2:
    #outlines
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Press space-bar to record position for recognition.", (10, 55), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "The more recorded positions, the more accurate the recognition will be.", (10, 80), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "For keybinds that can be executed ambidextrously, please record positions for both hands.", (10, 105), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "hands.", (10, 130), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Press 'Esc' to exit after recording.", (10, 155), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        #actual text
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "Press space-bar to record position for recognition.", (10, 55), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "The more recorded positions, the more accurate the recognition will be.", (10, 80), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "For keybinds that can be executed ambidextrously, please record positions for both", (10, 105), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "hands.", (10, 130), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "Press 'Esc' to exit after recording.", (10, 155), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
    else:
        #outlines
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Press 'Esc' to exit.", (10, 55), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv.LINE_AA)
        #actual text
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image, "Press 'Esc' to exit.", (10, 55), cv.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point']
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