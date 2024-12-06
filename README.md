pip install -r requirements.txt
you may need to python3 -m pip install mediapipe
or py -m pip install mediapipe

make sure you have a webcam and no other app is using it at the moment
then run app.py (click play button or ctrl f5)


if you have multiple python versions, use 3.11 beacuse mediapipe is stable here:
py -3.11 -m pip install -r requirements.txt
py -3.11 app.py

to custom train:
start app, anyhow select the 6 gestures, click proceed
the video feed should appear
press k and you will see "keypoint logging mode"
press 1,2,3,4,5,6,7,8,9,0 to train a total of 10 gestures
make sure to set the gesture names in keypoint_classifier/keypoint_classifier_label.csv
press q to quit
run learn_custom_gesture_keypoint.ipynb to train the model
then run app.py to test the model