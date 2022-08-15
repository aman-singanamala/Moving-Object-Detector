import cv2
car_classifier= cv2.CascadeClassifier("cars.xml")

vid= cv2.VideoCapture("video.mp4")

while True:
    ret , frame = vid.read()
    if ret== False:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    car = car_classifier.detectMultiScale(gray,1.04985,6)

    
    for (x,y,w,h) in car:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        

        cv2.imshow("Car detection", frame)


   

    key = cv2.waitKey(30)
    if(key==27):
        break



vid.release()
vid.destroyAllWindows()