import cv2
import mediapipe as mp

# Initialize the face and hand detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally to correct the mirrored view
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw a rectangle around the faces and add text annotation
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Add text 'Face'
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Convert the frame to RGB for hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform hand detection
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks and connections
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get the handedness (right or left hand)
            hand_label = results.multi_handedness[idx].classification[0].label
            
            # Correct handedness label if necessary
            if hand_label == 'Left':
                hand_label = 'Left'
            elif hand_label == 'Right':
                hand_label = 'Right'
            else:
                hand_label = 'Unknown'

            for landmark in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            # Draw landmarks and connections on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Add label to indicate right or left hand
            hand_position = (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h) - 20)
            cv2.putText(frame, hand_label, hand_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Face and Hand Detection", frame)
    
    # Break the loop if 's' key is pressed
    if cv2.waitKey(1) == ord("s"):
        break

cap.release()
cv2.destroyAllWindows()
