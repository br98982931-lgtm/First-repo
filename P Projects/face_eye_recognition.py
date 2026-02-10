# import cv2
# face_cap=cv2.CascadeClassifier("C:/Users/BHARGAV-RADADIYA/Downloads/haarcascade_frontalface_default.xml")
# eye_cap=cv2.CascadeClassifier("C:/Users/BHARGAV-RADADIYA/Downloads/haarcascade_eye.xml")
# # camera ne enable kare
# video_capture = cv2.VideoCapture(0)
# while True:
#     ret,video_data=video_capture.read()
#     # gray color ma convert
#     gray=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
#     # face detect karva
#     faces=face_cap.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30,30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#     # for loop thi box bane
#     for (x,y,w,h) in faces:
#         cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = video_data[y:y+h, x:x+w]

#         # Eye detection
#         eyes = eye_cap.detectMultiScale(
#             roi_gray,
#             scaleFactor=1.1,
#             minNeighbors=10
#         )
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color,(ex, ey),(ex + ew, ey + eh),(255, 0, 0),2)


#     cv2.imshow("my video",video_data)
#     # close mate Q press karo
#     if cv2.waitKey(10)==ord("q"):        
#         break 
# video_capture.release()

import cv2
import face_recognition
import pickle
import sqlite3
from datetime import datetime

# Load encodings
with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Connect database 
conn = sqlite3.connect("attendance.db")
cur = conn.cursor()

# Camera
video_capture = cv2.VideoCapture(0)

marked_names = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Attendance only once per session
            if name not in marked_names:
                marked_names.append(name)

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                cur.execute(
                    "INSERT INTO attendance(name, date, time) VALUES (?, ?, ?)",
                    (name, date, time)
                )
                conn.commit()

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("AI Attendance System", frame)

    if cv2.waitKey(10) == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
conn.close()
