from time import time
import cv2
import numpy as np
from collections import deque

# To quit once running, press 'Q' key while window is in focus.

# --- Settings ---
threshold = 10             # Filters out noise
amplification = 10         # Boosts visible motion
gaussianBlur = 3           # Smooths input frames
adaptiveThreshold = True   # Use mean-based threshold
fpsSmoothing = 0.9         # Smooths FPS readout

# Number of frames to compare against
# Higher = Higher degree of relation but, more 'Moton Blur'
# Lower  = Loses slow movement between frames but retains quality when high movement / frame
stackSize = 5

# --- Camera ---
# Set this to whatever your camera is set for. typically 60fps
desiredFPS = 60
# Camera 0 is the first index, if using multiple softwares, this may be different for you
# Typical Testing is run the app, if no video then increment this by one and retry
videoCapture = cv2.VideoCapture(0)
videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
videoCapture.set(cv2.CAP_PROP_FPS, desiredFPS)

# --- Initial frame + timing ---
previousTime = time()
fps = desiredFPS
frameStack = deque(maxlen=stackSize)
ret, previousFrame = videoCapture.read()
previousGray = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2GRAY)

# blur to remove noise
if gaussianBlur > 0:
    previousGray = cv2.GaussianBlur(previousGray, (gaussianBlur, gaussianBlur), 0)

# Main looop: gets newest frame and references previous saved frames for combination and subsequent modulization
while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    # Timing - fps smoothing
    currentTime = time()
    deltaTime = currentTime - previousTime
    previousTime = currentTime
    fps = fps * fpsSmoothing + (1.0 / deltaTime) * (1.0 - fpsSmoothing)

    # Blur to remove noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gaussianBlur > 0:
        gray = cv2.GaussianBlur(gray, (gaussianBlur, gaussianBlur), 0)

    # add to render stack
    frameStack.append(gray)

    # Add up diffs from past frames
    diffTotal = np.zeros_like(gray, dtype=np.float32)
    for past in frameStack:
        diffTotal += cv2.absdiff(gray, past).astype(np.float32)

    diffTotal /= len(frameStack)
    diffAmplified = cv2.convertScaleAbs(diffTotal * amplification)

    # Choose threshold
    if adaptiveThreshold:
        thresholdValue = max(threshold, np.mean(diffTotal) * 1.5)
    else:
        thresholdValue = threshold

    # Build motion mask
    _, motionMask = cv2.threshold(diffAmplified, thresholdValue, 255, cv2.THRESH_BINARY)
    motionOnly = cv2.bitwise_and(diffAmplified, diffAmplified, mask=motionMask)
    output = cv2.merge([motionOnly] * 3)

    # HUD
    cv2.putText(output, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("MotionAccumulator", output)

    # Q to Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# on close
videoCapture.release()
cv2.destroyAllWindows()
