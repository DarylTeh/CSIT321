import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import win32api
import win32con
import tkinter as tk
from tkinter import ttk

import time
import asyncio

import cv2 as cv
import numpy as np
import mediapipe as mp

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

frameList={}

def navigateTo(page):
    if page in frameList:
        frame = frameList[page]
        frame.tkraise()
        print(f"Frame {page} get loaded")
    else:
        print(f"Page {page} not found.")

class Root(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Main Menu")
        self.geometry("500x500")
        self.configure(bg="#f0f0f0")
        
        mainFrame = Frame(self)
        mainFrame.pack(fill="both", expand=True)
        
        mainFrame.grid_rowconfigure(0, weight=1)
        mainFrame.grid_columnconfigure(0, weight=1)
        
        # self.frames = {}
        
        for page in (mainMenuUI, predefinedHandGesturesUI, customHandGesturesUI):
            print(f"Initializing frame for {page.getIdentity()}")
            frame = page(mainFrame, self)
            frameList[page.getIdentity()] = frame
            frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
        print("frames count: {0}".format(str(len(frameList))))
            
        for x in frameList:
            print("frames item: {0}".format(x))
        
        navigateTo(MAINMENU_UI)

class mainMenuUI(ttk.Frame):
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        title = Label(self, text=mainMenuTitle, font=titleSize)
        title.pack()
        
        # predefinedHandGesturesBtn = buildButton(self, "Predefined Hand Gestures", lambda: controller.navigateTo(PREDEFINED_HG_UI))
        predefinedHandGesturesBtn = buildButton(self, "Predefined Hand Gestures", PREDEFINED_HG_UI)
        predefinedHandGesturesBtn.pack(padx=20, pady=20)
        
        customHandGesturesBtn = buildButton(self, "Custom Hand Gestures", CUSTOM_HG_UI)
        customHandGesturesBtn.pack(padx=20, pady=20)
        
        testingBtn = buildButton(self, "Test Hand Gestures", TESTING_HG_UI)
        testingBtn.pack(padx=20, pady=20)
        
        # startGameBtn = buildButton(self, "Start Game", lambda: controller.navigateTo())
        # startGameBtn.pack(padx=20, pady=20)
        
    def getIdentity():
        return MAINMENU_UI
        
class predefinedHandGesturesUI(ttk.Frame):
    
    def __init__(self, parent, controller):
        super().__init__(parent, padding=20)
        self.controller = controller
        
        title = Label(self, text=predefinedHGTitle, font=titleSize)
        title.pack()
        
        # keyboardBtn = buildButton(self, "Keyboard", )
        # mouseBtn = buildButton(self, "Mouse", )
        doneBtn = buildButtonWithColor(self, "Done", MAINMENU_UI, "green")
        
        # keyboardBtn.pack(padx=20, pady=20)
        # mouseBtn.pack(padx=20, pady=20)
        doneBtn.pack(padx=20, pady=20)
        
    def getIdentity():
        return PREDEFINED_HG_UI
        

class customHandGesturesUI(ttk.Frame):
    
    def __init__(self, parent, controller):
        super().__init__(parent, padding=20)
        self.controller = controller
        
        title = Label(self, text=customHGTitle, font=titleSize)
        title.pack()
        
        # keyboardBtn = buildButton(self, "Keyboard", )
        # mouseBtn = buildButton(self, "Mouse", )
        doneBtn = buildButtonWithColor(self, "Done", MAINMENU_UI, "green")
        
        # keyboardBtn.pack(padx=20, pady=20)
        # mouseBtn.pack(padx=20, pady=20)
        doneBtn.pack(padx=20, pady=20)
        
    def getIdentity():
        return CUSTOM_HG_UI  
        
class testingHGUI(ttk.Frame):
    
    def __init__(self, parent, controller):
        super().__init__(parent, padding=20)
        self.controller = controller
        
        title = Label(self, text=customHGTitle, font=titleSize)
        title.pack()
        
        # keyboardBtn = buildButton(self, "Keyboard", )
        # mouseBtn = buildButton(self, "Mouse", )
        # doneBtn = buildButton(self, "Done", , "green")
        
        # keyboardBtn.pack(padx=20, pady=20)
        # mouseBtn.pack(padx=20, pady=20)
        # doneBtn.pack(padx=20, pady=20)
        
    def getIdentity():
        return TESTING_HG_UI
        
def buildButton(frame, text, pageName):
    
    button = Button(
        frame,
        text=text,
        command= lambda: navigateTo(pageName),
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

def buildButtonWithColor(frame, text, pageName, color):
    
    button = Button(
        frame,
        text=text,
        command= lambda: navigateTo(pageName),
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

async def press_key_throttled(hex_key_code):
    global last_key_press_time
    current_time = time.time()
    # Check if the time since the last keypress is greater than the throttle time.
    if current_time - last_key_press_time >= THROTTLE_TIME:
        win32api.keybd_event(hex_key_code, 0, 0, 0)
        time.sleep(0.2)  # Short delay to ensure the key press is registered
        # Simulates releasing the key
        win32api.keybd_event(hex_key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
        last_key_press_time = current_time

# def is_thumb_up(landmarks):
#     # Thumb up: Thumb straight, other fingers clenched
#     thumb_tip = landmarks[4]
#     thumb_base = landmarks[1]
    
#     # Check if thumb is straight up (x, y coordinate check based on the angle of thumb)
#     is_thumb_straight = thumb_tip.y < thumb_base.y and abs(thumb_tip.x - thumb_base.x) < 0.1
    
# #     # Check if other fingers are clenched
# #     are_other_fingers_clenched = (
# #         landmarks[8].y > landmarks[6].y and  # Index finger clenched
# #         landmarks[12].y > landmarks[10].y and  # Middle finger clenched
# #         landmarks[16].y > landmarks[14].y and  # Ring finger clenched
# #         landmarks[20].y > landmarks[18].y    # Pinky clenched
# #     )
    
#     return is_thumb_straight and are_other_fingers_clenched

# def is_thumb_down(landmarks):
#     # Thumb down: Thumb straight, other fingers clenched
#     thumb_tip = landmarks[4]
#     thumb_base = landmarks[1]
    
#     # Check if thumb is straight down (x, y coordinate check based on the angle of thumb)
#     is_thumb_straight = thumb_tip.y > thumb_base.y and abs(thumb_tip.x - thumb_base.x) < 0.1
    
#     # Check if other fingers are clenched
#     are_other_fingers_clenched = (
#         landmarks[8].y > landmarks[6].y and  # Index finger clenched
#         landmarks[12].y > landmarks[10].y and  # Middle finger clenched
#         landmarks[16].y > landmarks[14].y and  # Ring finger clenched
#         landmarks[20].y > landmarks[18].y    # Pinky clenched
#     )
    
#     return is_thumb_straight and are_other_fingers_clenched

# def is_middle_finger_up(landmarks):
#     # Middle finger up: Middle finger raised, others clenched
#     middle_finger_tip = landmarks[12]
#     ring_finger_tip = landmarks[16]
#     index_finger_tip = landmarks[8]
    
#     # Check if middle finger is raised
#     is_middle_raised = middle_finger_tip.y < ring_finger_tip.y and middle_finger_tip.y < index_finger_tip.y
    
#     # Check if other fingers are clenched
#     are_other_fingers_clenched = (
#         landmarks[8].y > landmarks[6].y and  # Index finger clenched
#         landmarks[16].y > landmarks[14].y and  # Ring finger clenched
#         landmarks[20].y > landmarks[18].y    # Pinky clenched
#     )
    
#     return is_middle_raised and are_other_fingers_clenched

# def is_peace_sign(landmarks):
#     # Peace sign (V sign): Index and middle finger raised, others clenched
#     index_finger_tip = landmarks[8]
#     middle_finger_tip = landmarks[12]
#     ring_finger_tip = landmarks[16]
#     pinky_tip = landmarks[20]

#     # Check if index and middle fingers are raised
#     are_index_and_middle_raised = index_finger_tip.y < landmarks[6].y and middle_finger_tip.y < landmarks[10].y
    
#     # Check if ring and pinky fingers are clenched
#     are_ring_and_pinky_clenched = (
#         ring_finger_tip.y > landmarks[14].y and  # Ring finger clenched
#         pinky_tip.y > landmarks[18].y            # Pinky clenched
#     )
    
#     return are_index_and_middle_raised and are_ring_and_pinky_clenched

# def is_ok_sign(landmarks):
#     # OK sign: Thumb and index finger touch to form a circle, other fingers clenched
#     thumb_tip = landmarks[4]
#     index_finger_tip = landmarks[8]
#     thumb_index_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
    
#     # Check if thumb and index finger are touching (distance between tips is small)
#     is_thumb_index_touching = thumb_index_distance < 0.05
    
#     # Check if other fingers are clenched
#     are_other_fingers_clenched = (
#         landmarks[12].y > landmarks[10].y and  # Middle finger clenched
#         landmarks[16].y > landmarks[14].y and  # Ring finger clenched
#         landmarks[20].y > landmarks[18].y      # Pinky clenched
#     )
    
#     return is_thumb_index_touching and are_other_fingers_clenched

# def is_fist(landmarks):
#     # Fist: All fingers clenched, thumb across the palm
#     thumb_tip = landmarks[4]
#     index_finger_tip = landmarks[8]
#     middle_finger_tip = landmarks[12]
#     ring_finger_tip = landmarks[16]
#     pinky_tip = landmarks[20]
    
#     # Check if all fingers are clenched
#     are_all_fingers_clenched = (
#         index_finger_tip.y > landmarks[6].y and  # Index finger clenched
#         middle_finger_tip.y > landmarks[10].y and  # Middle finger clenched
#         ring_finger_tip.y > landmarks[14].y and  # Ring finger clenched
#         pinky_tip.y > landmarks[18].y            # Pinky clenched
#     )
    
#     # Check if thumb is across the palm
#     is_thumb_clenched = thumb_tip.x < landmarks[3].x  # Thumb crosses inward
    
#     return are_all_fingers_clenched and is_thumb_clenched


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
    def proceed():
        print("Gesture to key mapping:")
        for gesture, hex_code in gesture_to_key.items():
            print(f"{gesture}: {hex_code}")
        # Here you can add any additional actions for when the proceed button is pressed
        root.quit()

    # Create the main window
    # root = tk.Tk()
    # root.title("Main Menu")
    # root.geometry("500x500")
    # root.configure(bg="#f0f0f0")
    root = Root()

    # Use a nicer theme
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Arial', 12), padding=5)
    style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='black')
    style.configure('TLabel', font=('Arial', 12), background='#f0f0f0')
    

    # Create a frame to hold all mappings
    # frame = ttk.Frame(root, padding=20, style='Frame.TFrame')
    # frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


    # Header Label
    # header_label = ttk.Label(frame, text="Map Gestures to Key Presses", style='Header.TLabel')
    # header_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    # # Add labels and buttons for each gesture
    # for idx, gesture in enumerate(gestures):
    #     label = ttk.Label(frame, text=f"{gesture}: Not assigned", foreground="red")
    #     label.grid(row=idx + 1, column=0, padx=5, pady=10, sticky=tk.W)
        
    #     button = ttk.Button(frame, text="Click to record", 
    #                         command=lambda g=gesture, l=label: enable_recording(g, l))
    #     button.grid(row=idx + 1, column=1, padx=10, pady=10)
        
    #     # Add hover effect to buttons
    #     def on_enter(event, b=button):
    #         b.config(style='Hover.TButton')
        
    #     def on_leave(event, b=button):
    #         b.config(style='TButton')

    #     button.bind("<Enter>", on_enter)
    #     button.bind("<Leave>", on_leave)

    # style.configure('Hover.TButton', background='#80c1ff', font=('Arial', 12, 'bold'))

    # # Add a proceed button
    # proceed_button = ttk.Button(frame, text="Proceed", command=proceed, style='TButton')
    # proceed_button.grid(row=len(gestures) + 2, column=0, columnspan=2, pady=20)

    # Run the application
    root.mainloop()


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
            
            print("Total Thumbs Down Count : "+str(thumbDownCount))
            thumbDownAlgo.printHighestDistanceGroupingStatistic()
            print("Total Thumbs Up Count : "+str(thumbUpCount))
            thumbUpAlgo.printHighestDistanceGroupingStatistic()
            print("Total Close Count : "+str(closeCount))
            closeAlgo.printHighestDistanceGroupingStatistic()
            print("Total OK Count : "+str(okayCount))
            okayAlgo.printHighestDistanceGroupingStatistic()
            
            break
        number, mode = select_mode(key, mode)

        # means if the camera capture something, then ret is True. Else ret is False.
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
                
                if is_middle_finger_up(hand_landmarks.landmark):
                    gesture_text = "Middle Finger"
                    print("Fuck youuuu")
                    # printDistanceBetweenLandmarks(hand_landmarks.landmark)
                elif is_close(hand_landmarks.landmark):
                    gesture_text = "Close"
                    closeCount += 1
                    print("Close")
                    closeAlgo.getLandMarkWidthAndHeightDistanceOfOneGestureAllFingers(hand_landmarks.landmark)
                    printDistanceBetweenLandmarks(hand_landmarks.landmark)
                elif is_thumb_up(hand_landmarks.landmark):
                    gesture_text = "Thumb Up"
                    print("Thumb up")
                    thumbUpCount += 1
                    printDistanceBetweenLandmarks(hand_landmarks.landmark)
                    thumbUpAlgo.getLandMarkWidthAndHeightDistanceOfOneGestureAllFingers(hand_landmarks.landmark)
                    
                elif is_thumb_down(hand_landmarks.landmark):
                    gesture_text = "Thumb Down"
                    thumbDownCount += 1
                    print("Thumb down")
                    thumbDownAlgo.getLandMarkWidthAndHeightDistanceOfOneGestureAllFingers(hand_landmarks.landmark)
                    printDistanceBetweenLandmarks(hand_landmarks.landmark)
                elif is_peace_sign(hand_landmarks.landmark):
                    gesture_text = "Peace Sign"
                    print("RIP")
                    # printDistanceBetweenLandmarks(hand_landmarks.landmark)
                elif is_ok_sign(hand_landmarks.landmark):
                    gesture_text = "OK Sign"
                    okayCount += 1
                    print("Ogay")
                    printDistanceBetweenLandmarks(hand_landmarks.landmark)
                    okayAlgo.getLandMarkWidthAndHeightDistanceOfOneGestureAllFingers(hand_landmarks.landmark)
                elif is_fist(hand_landmarks.landmark):
                    gesture_text = "Fist"
                    print("Bagelo")
                    printDistanceBetweenLandmarks(hand_landmarks.landmark)
                else:
                    gesture_text = keypoint_classifier_labels[hand_sign_id]
                    print("Unknown: "+gesture_text)
                    printDistanceBetweenLandmarks(hand_landmarks.landmark)

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    gesture_text,
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
                
                # print("gesture_text : "+gesture_text)

                if (is_middle_finger_up(hand_landmarks.landmark) or
                    is_thumb_up(hand_landmarks.landmark) or
                    is_thumb_down(hand_landmarks.landmark) or
                    is_peace_sign(hand_landmarks.landmark) or
                    is_ok_sign(hand_landmarks.landmark) or
                    is_fist(hand_landmarks.landmark)):
                    asyncio.run(press_key_throttled(gesture_to_key[gesture_text]))
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('HGR To Play Games', debug_image)

    cap.release()
    cv.destroyAllWindows()


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