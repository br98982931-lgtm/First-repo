import cv2
face_cap=cv2.CascadeClassifier("C:/Users/BHARGAV-RADADIYA/Downloads/haarcascade_frontalface_default.xml")
# camera ne enable kare
video_capture = cv2.VideoCapture(0)
while True:
    ret,video_data=video_capture.read()
    # gray color ma convert
    col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
    # face detect karva
    faces=face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # for loop thi box bane
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("my video",video_data)
    if cv2.waitKey(10)==ord("q"):
        break 
video_capture.release()
 