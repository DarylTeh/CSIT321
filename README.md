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
• improve recognition in low light, slight variation conditions         (almost impossible unless we finetnue mediapipe)
*****our own classifier if have time & optimise model inference time    (stochastic gradient desc + momentum + RMSprop)
