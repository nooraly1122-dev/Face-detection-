import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os

# Load trained model and labels
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

with open('labels.pickle', 'rb') as f:
    label_dict = pickle.load(f)
    # Reverse mapping: ID -> Name
    id_to_name = {v: k for k, v in label_dict.items()}

# Initialize face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Attendance CSV setup
attendance_file = 'attendance.csv'

# Create file with headers if it doesn't exist
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    df.to_csv(attendance_file, index=False)

# Keep track of already marked today
marked_today = set()

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    
    # Check if already marked today
    if name in marked_today:
        return False
    
    # Read existing attendance
    df = pd.read_csv(attendance_file)
    
    # Check if already in CSV for today
    if not df[(df['Name'] == name) & (df['Date'] == today)].empty:
        marked_today.add(name)
        return False
    
    # Add new record
    new_entry = pd.DataFrame([[name, today, now]], columns=['Name', 'Date', 'Time'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    marked_today.add(name)
    return True

# Start webcam
video_cap = cv2.VideoCapture('http://localhost:4747')
print("=" * 50)
print("FACE RECOGNITION ATTENDANCE SYSTEM")
print("=" * 50)
print("Press 'q' to quit")
print("Looking for faces...")
print("-" * 50)

while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Predict face
        id_pred, conf = recognizer.predict(roi_gray)
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Check confidence (lower is better for LBPH)
        if conf < 90:  # Good match (zyada flexible)
            name = id_to_name[id_pred]
            
            # Mark attendance
            if mark_attendance(name):
                print(f"✓ Attendance marked for {name} at {datetime.now().strftime('%H:%M:%S')}")
                cv2.putText(frame, f"MARKED: {name}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"{name} (Already Marked)", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            name = "Unknown"
            cv2.putText(frame, "UNKNOWN", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Face Recognition Attendance System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
print("-" * 50)
print("System closed")

# Show today's attendance
print("\n--- TODAY'S ATTENDANCE REPORT ---")
df = pd.read_csv(attendance_file)
today = datetime.now().strftime("%Y-%m-%d")
today_attendance = df[df['Date'] == today]
if today_attendance.empty:
    print("No attendance recorded today")
else:
    print(today_attendance.to_string(index=False))
print("-" * 50)