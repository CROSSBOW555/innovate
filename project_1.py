import face_recognition
import cv2
import numpy as np 
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known face encodings and names
mohit_image = face_recognition.load_image_file("IMAGES\\mohit.jpg")
mohit_encoding = face_recognition.face_encodings(mohit_image)[0]
elon_image = face_recognition.load_image_file("IMAGES\\elon.png")
elon_encoding = face_recognition.face_encodings(elon_image)[0]
enola_image = face_recognition.load_image_file("IMAGES\\enola.png")
enola_encoding = face_recognition.face_encodings(enola_image)[0]
modi_image = face_recognition.load_image_file("IMAGES\\modi.png")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

know_face_encoding = [
    mohit_encoding,
    elon_encoding,
    enola_encoding,
    modi_encoding
]

know_face_names = [
    "Mohit",
    "elon",
    "enola",
    "modi"
]

students = know_face_names.copy()

# Get current date for file name
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open CSV file for writing attendance
with open(current_date + '.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    
    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(know_face_encoding, face_encoding)
            name = ""
            face_distances = face_recognition.face_distance(know_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = know_face_names[best_match_index]
                face_names.append(name)
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
