import cv2
import os

video_path = 'me.mp4'
output_dir = 'dataset/mine'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))
        filename = os.path.join(output_dir, f"me_{saved}.jpg")
        cv2.imwrite(filename, face)
        saved += 1
        break  # save only one face per frame

    frame_count += 1

cap.release()
print(f"✅ Done: Extracted {saved} face images from video!")
