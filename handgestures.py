import cv2
import mediapipe as mp
import pyttsx3
import math

# Initialize TTS (optional)
engine = pyttsx3.init()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Finger tip landmark indices
FINGER_TIPS = [8, 12, 16, 20]
THUMB_TIP = 4

def get_finger_status(hand_landmarks):
    finger_status = []

    # Thumb (check x instead of y)
    if hand_landmarks.landmark[THUMB_TIP].x < hand_landmarks.landmark[THUMB_TIP - 2].x:
        finger_status.append(1)
    else:
        finger_status.append(0)

    # Other fingers
    for tip in FINGER_TIPS:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_status.append(1)
        else:
            finger_status.append(0)

    return finger_status

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def recognize_gesture(status, landmarks):
    # Fist
    if status == [0, 0, 0, 0, 0]:
        return "Fist"

    # Open Palm
    elif status == [1, 1, 1, 1, 1]:
        return "Open Palm"

    # One Finger
    elif status == [0, 1, 0, 0, 0]:
        return "One Finger"

    # Peace
    elif status == [0, 1, 1, 0, 0]:
        return "Peace"

    # Thumbs Up
    elif status == [1, 0, 0, 0, 0]:
        return "Thumbs Up"

    # Call Me (Thumb + Pinky)
    elif status == [1, 0, 0, 0, 1]:
        return "Call Me"

    # Pointing (Index Only)
    elif status == [0, 1, 0, 0, 0]:
        return "Pointing"

    # OK Sign: Thumb tip close to index tip
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    d = distance(thumb_tip, index_tip)
    if d < 0.05:  # threshold can be tuned
        return "OK"

    return "Unknown"

# Start webcam
cap = cv2.VideoCapture(0)

last_gesture = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_status = get_finger_status(hand_landmarks)
            gesture = recognize_gesture(finger_status, hand_landmarks)


            if gesture != last_gesture:
                last_gesture = gesture
                print("Gesture:", gesture)
                engine.say(gesture)
                engine.runAndWait()

            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
