import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

tripti_image = face_recognition.load_image_file("IMAGES/tripti.jpg")
tripti_encoding = face_recognition.face_encodings(tripti_image)[0]

mohit_image = face_recognition.load_image_file("IMAGES/mohit.jpg")
mohit_encoding = face_recognition.face_encodings(mohit_image)[0]

enola_image = face_recognition.load_image_file("IMAGES/enola.png")
enola_encoding = face_recognition.face_encodings(enola_image)[0]

elon_image = face_recognition.load_image_file("IMAGES/elon.png")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

modi_image = face_recognition.load_image_file("IMAGES/modi.png")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

shubham_image = face_recognition.load_image_file("IMAGES/shubham.jpg")
shubham_encoding = face_recognition.face_encodings(shubham_image)[0]

HARSH_image = face_recognition.load_image_file("IMAGES/HARSH.jpg")
HARSH_encoding = face_recognition.face_encodings(HARSH_image)[0]

piyush_image = face_recognition.load_image_file("IMAGES/piyush.jpg")
piyush_encoding = face_recognition.face_encodings(piyush_image)[0]

priya_image = face_recognition.load_image_file("IMAGES/priya.jpg")
priya_encoding = face_recognition.face_encodings(priya_image)[0]


known_face_encoding = [
    HARSH_encoding,
    piyush_encoding,
    shubham_encoding,
    tripti_encoding,
    mohit_encoding,
    enola_encoding,
    elon_encoding,
    modi_encoding,
    priya_encoding
]

known_faces_names = [
    "HARSH",
    "piyush",
    "shubham",
    "tripti",
    "mohit",
    "enola",
    "elon",
    "modi",
    "priya"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(
                known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                bottomLeftCornerOfText = (10, 100)
                fontScale = 2
                fontColor = (240, 120, 60)
                thickness = 2
                lineType = 3

                cv2.putText(frame, name+' Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
