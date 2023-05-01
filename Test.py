
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")
labels = ["B", "Thumbsup", "Y", "Yoo"]

offset = 20
imgSize = 300


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
          #  print (prediction, index)


        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw= False)
          #  print(prediction, index)

        cv2.putText((imgOutput), labels[index],(x,y-29), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset, y+h+offset),(0,0,0),4)


     #   cv2.imshow("ImageWhite", imgWhite)

    cv2.rectangle(imgOutput,(0,0),(800,50),(145,169,43),cv2.FILLED)
   # cv2.putText((imgOutput),"Project by Team 2", (0,15), cv2.FONT_ITALIC, 0.8, (0, 0, 0), 2)
    cv2.putText((imgOutput),"SIGN LANGUAGE DETECTOR" , (80,35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)














