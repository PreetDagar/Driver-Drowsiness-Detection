import time
import threading
import cv2
import imutils
import dlib
from scipy.spatial import distance
from imutils import face_utils
from playsound import playsound  # Import playsound for playing sound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alert_sound():
    print("Playing sound...")
    playsound('alert.wav')  # Ensure this is the correct path to the sound file

# Setup the face detector and predictor
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0
alarm_on = False

# Initialize time tracking variables
start_time = None  # Start time should be initialized once the alert is triggered

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1

            if flag >= frame_check:
                # Start tracking time only once the alert is triggered
                if start_time is None:
                    start_time = time.time()

                elapsed_time = time.time() - start_time
                time_left = int(elapsed_time)  # Display time in seconds

                # Add visual feedback for drowsiness
                cv2.rectangle(frame, (0, 0), (450, 50), (0, 0, 255), -1)
                cv2.putText(frame, f"ALERT! {time_left} sec", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Play sound only once
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alert_sound).start()

        else:
            flag = 0
            alarm_on = False
            start_time = None  # Reset start time when drowsiness is not detected

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
