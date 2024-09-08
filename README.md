## Face and Hand Detection using OpenCV and Mediapipe

### Features:
- Detects faces using OpenCV's Haar Cascade classifier.
- Detects hands and their landmarks using MediaPipe's hand detection module.
- Displays rectangles around detected faces with text annotation ("Face").
- Draws hand landmarks and connections between them.
- Runs in real-time using your webcam.

### Requirements:
- Python 3.x
- OpenCV
- Mediapipe

### How to Use:
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install opencv-python mediapipe
   ```
3. Run the script:
   ```bash
   python face_hand_detection.py
   ```
4. Press the 's' key to stop the program.

### Notes:
- The script uses the default webcam for video input.
- Adjust the detection confidence as needed.
