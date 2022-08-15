from errno import EUCLEAN
from turtle import width
import cv2
cap= cv2.VideoCapture("video.mp4")
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from tracker import *
# object detection
# extracts the moving objects from the stable camera

object_detector=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)




while True:
    ret , frame = cap.read()
    # mask is for detecting the moving objects


    # Extract Region of interest
    height, width,_ =frame.shape
    #print(height,width)
    roi= frame[300:800,600:1500]


    mask= object_detector.apply(roi)
    # clean the mask ( to detect only objects ot their shadows)
    _, mask= cv2.threshold(mask,254,255,cv2.THRESH_BINARY) # we reomve pixels of color 254 below, keep white 
    # object detection
    contours, _=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in contours:

        
        #Remove elements based on the area
        area=cv2.contourArea(i)
        if area >100:
            #cv2.drawContours(roi,[i],-1,(0,255,0),2)
            x,y,w,h= cv2.boundingRect(i)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)




    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",mask)
    cv2.imshow("Roi",roi)
    key = cv2.waitKey(30)
    if(key==27):
        break



cap.release()
cap.destroyAllWindows()