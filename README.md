pip install -r requirements.txt
you may need to python3 -m pip install mediapipe
or py -m pip install mediapipe

make sure you have a webcam and no other app is using it at the moment
then run app.py (click play button or ctrl f5)


if you have multiple python versions, use 3.11 beacuse mediapipe is stable here:
py -3.11 -m pip install -r requirements.txt
py -3.11 app.py




TODO
• loading throbber popup                                                (delete custom gesture done, still need for adding/retraining/starting cam)
• mouse capability                                                      (win32con + ownself bounding box tracking algo)
• fix UI elements
• make custom training easier
• allow arrow keys/special keys (win32con)
• allow combo gestures to be set as keystroke                           (might not do)
• change key hex code in key binds menu to actual key names             (done for keyboard hexcodes, not mouse)
• improve recognition in low light, slight variation conditions         (almost impossible unless we finetune mediapipe)
*****our own classifier if have time & optimise model inference time    (stochastic gradient desc + momentum + RMSprop)


---------------------------------------------------------------------------------------------------------------------------------------------------------
Methods & Classes and what they do: 
buildModel - build new model arch; set layers
compileModel - compile the model with optimizer, loss func, Adam
createCheckpointCallback - in case error, still saves up till last successful iteration
trainModelWithCustomHandGesture - train using keypoint.csv
show_loading_popup - show the spinner popup (for add/delete)
hide_loading_popup - hide the spinner popup (for add/delete)
addHandGestureToCSV - adds a new gesture class into keypoint_classifier_label.csv
addNewCustomGesture - UI stuff
deleteCustomGesture - UI stuff
deleteHandGestureFromCSV - delete gesture class from keypoint_classifer_label.csv

class CustomHandGestureObject - ???
class Root - UI stuff
class MainMenuUI - UI stuff
class PredefinedHandGesturesUI
class CustomHandGesturesUI
class PredefinedHandGesturesComponent
class PredefinedHandGesturesKeyboardUI
class PredefinedHandGesturesMouseUI
class CustomHandGesturesKeyboardUI
class NewCustomHandGestureForKeyboard
class TestingHGUI

press_key_turbo - for holding down key
press_key_throttled - for tapping of key
initiateWebCam - start camera + mediapipe + cv
get_args - get camera, frame width, static img mode(means only keypoint, no history)
main - start the tkinter Root UI
hand_cursor_control - move cursor to translated cords using index fingertip relative to frame
calc_bounding_rect - calculate corners of bounding box
calc_landmark_list - normalise to pixel, easier to draw in frame
pre_process_landmark - normalise to relative cords, makes data same scale, ignores absolute position
pre_process_point_history - normalise to relative coords, makes data same scale & resolution
logging_csv - append new cords to keypoint.csv
draw_landmarks - UI stuff; 21 landmark dots on each hand
draw_bounding_rect - UI stuff; rectangle around each hand
draw_info_text - UI stuff; gesture
draw_point_history - UI stuff; index fingertip trailing circles
draw_info - UI stuff; fps, logging label