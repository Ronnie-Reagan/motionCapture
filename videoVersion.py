from time import time
import cv2
import numpy as np
from collections import deque
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# --- Video Selection ---
Tk().withdraw()  # Hide the root tkinter window
videoPath = askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
if not videoPath:
    print("No file selected. Exiting.")
    exit()

# --- Settings ---
threshold = 10             # Filters out noise
amplification = 10         # Boosts visible motion
gaussianBlur = 3           # Smooths input frames
adaptiveThreshold = True   # Use mean-based threshold
fpsSmoothing = 0.9         # Smooths FPS readout
stackSize = 5             # Number of frames to compare against

# --- Video ---
videoCapture = cv2.VideoCapture(videoPath)
if not videoCapture.isOpened():
    print(f"Failed to open video: {videoPath}")
    exit()

frameStack = deque(maxlen=stackSize)
fps = videoCapture.get(cv2.CAP_PROP_FPS) or 60
previousTime = time()

# --- First Frame ---
ret, previousFrame = videoCapture.read()
if not ret:
    print("Failed to read first frame.")
    exit()

previousGray = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2GRAY)
if gaussianBlur > 0:
    previousGray = cv2.GaussianBlur(previousGray, (gaussianBlur, gaussianBlur), 0)

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    # Timing
    currentTime = time()
    deltaTime = currentTime - previousTime
    previousTime = currentTime
    fps = fps * fpsSmoothing + (1.0 / deltaTime) * (1.0 - fpsSmoothing)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gaussianBlur > 0:
        gray = cv2.GaussianBlur(gray, (gaussianBlur, gaussianBlur), 0)

    frameStack.append(gray)

    diffTotal = np.zeros_like(gray, dtype=np.float32)
    for past in frameStack:
        diffTotal += cv2.absdiff(gray, past).astype(np.float32)
    diffTotal /= len(frameStack)
    diffAmplified = cv2.convertScaleAbs(diffTotal * amplification)

    if adaptiveThreshold:
        thresholdValue = max(threshold, np.mean(diffTotal) * 1.5)
    else:
        thresholdValue = threshold

    _, motionMask = cv2.threshold(diffAmplified, thresholdValue, 255, cv2.THRESH_BINARY)
    motionOnly = cv2.bitwise_and(diffAmplified, diffAmplified, mask=motionMask)
    output = cv2.merge([motionOnly] * 3)

    cv2.putText(output, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("MotionAccumulator", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
