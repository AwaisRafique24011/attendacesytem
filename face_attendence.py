import cv2
import numpy as np
import os
from datetime import datetime

# Load the pre-trained face detection model
face_cap = cv2.CascadeClassifier("C:/Users/Faiza/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Path to the folder containing reference images
path = 'images'
# the path mean in which you stored the previous image of students that you use for attendance.
images = []
classNames = []
face_encodings = []

# Load and encode images
for cls in os.listdir(path):
    img = cv2.imread(f'{path}/{cls}')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = img_rgb[y:y + h, x:x + w]
        # Resize face to a fixed size
        face_resized = cv2.resize(face, (100, 100))  # Resize to 100x100
        images.append(face_resized)
        classNames.append(os.path.splitext(cls)[0])

# Function to find the encoding of a face
def face_encode(image):
    # Resize and flatten the image to create a simple encoding
    image_resized = cv2.resize(image, (100, 100))  # Resize to 100x100
    return image_resized.flatten()

# Function to compare faces
def compare_faces(known_face_encodings, face_encoding):
    matches = []
    for known_face_encoding in known_face_encodings:
        match = np.linalg.norm(known_face_encoding - face_encoding)
        matches.append(match)
    return matches

# Function to mark attendance
def markAttendance(name):
    file_exists = os.path.isfile('Attendance.csv')
    
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')
        
        if not file_exists:
            f.write('Name,Time\n')

# Encode all loaded images
for img in images:
    encoding = face_encode(img)
    face_encodings.append(encoding)

# Track attendance for the current session
marked_attendance = set()

# Start video capture
video_cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, video_data = video_cap.read()

    # Check if frame is captured
    if not ret:
        break

    # Convert the frame to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Encode faces detected in the frame
    for (x, y, w, h) in faces:
        face = col[y:y + h, x:x + w]
        face_encoding = face_encode(face)
        matches = compare_faces(face_encodings, face_encoding)

        if len(matches) > 0 and min(matches) < 10000:  # Adjust threshold as needed
            name = classNames[np.argmin(matches)]

            if name not in marked_attendance:
                markAttendance(name)
                marked_attendance.add(name)
                cv2.putText(video_data, f'{name} - Attendance marked', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(video_data, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("video_live", video_data)

    # Break the loop if 'a' is pressed
    if cv2.waitKey(10) == ord('a'):
        break

# Release the capture and close windows
video_cap.release()
cv2.destroyAllWindows()
