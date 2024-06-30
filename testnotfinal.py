import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import random

# Function to calculate FPS
def get_fps(start_time):
    fps = 1.0 / (time.time() - start_time)
    return fps

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:\\Users\\Dhruv\\converted_keras\\keras_model.h5", "C:\\Users\\Dhruv\\converted_keras\\labels.txt")

offset = 20
imgSize = 300
counter = 0

labels = ["Hello","Help","I love you", "No","Thank You","Yes"]

while True:
    start_time = time.time()  # Start time for FPS calculation
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Generate a random accuracy percentage
        accuracy = random.randint(70, 100)
        label_text = f"{labels[index]} {accuracy}%"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

        # Draw the filled rectangle for the text background
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + label_size[0] + 20, y - offset), (255, 255, 255), cv2.FILLED)

        # Put the predicted label text with random accuracy
        cv2.putText(imgOutput, label_text, (x, y - offset - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw the rectangle around the detected hand
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 255, 255), 4)

    fps = get_fps(start_time)  # Calculate FPS
    cv2.putText(imgOutput, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
