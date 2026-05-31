import cv2
import os
import numpy as np
import pickle

# Create dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize camera - Try multiple connection methods
print("Connecting to DroidCam...")
video_cap = cv2.VideoCapture('http://localhost:4747')

if not video_cap.isOpened():
    print("USB connection failed. Trying WiFi...")
    video_cap = cv2.VideoCapture('http://192.168.100.97:4747')

if not video_cap.isOpened():
    print("WiFi connection failed. Trying default camera (0)...")
    video_cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not video_cap.isOpened():
    print("ERROR: Could not open any camera source.")
    print("Please check:")
    print("1. DroidCam app is open on your phone")
    print("2. USB cable is connected (or WiFi is enabled)")
    print("3. DroidCam shows 'Connected' status")
    exit()
    if not video_cap.isOpened():
        print("ERROR: Could not connect to DroidCam. Check:")
        print("1. DroidCam is running on your phone")
        print("2. IP address 192.168.100.97 is correct")
        print("3. Phone and PC are on same network")
        print("4. DroidCam wifi mode is enabled")
        exit()

print("✓ Camera connected successfully!")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get student info
student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

# Create student folder
student_folder = os.path.join('dataset', f"{student_id}_{student_name}")
if not os.path.exists(student_folder):
    os.makedirs(student_folder)

print("Looking at camera... Keep your face in frame")
print("Press 'q' to quit early")
count = 0

while count < 30:  # Capture 30 images per person
    ret, frame = video_cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1
        # Save the face image
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{student_folder}/img_{count}.jpg", face_img)
        
    cv2.putText(frame, f"Captured: {count}/30", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture Faces - Look at Camera", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Face collection complete! Captured {count} images")
video_cap.release()
cv2.destroyAllWindows()

# ------------------- TRAINING PART -------------------
print("Training model... Please wait")

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces_data = []
labels = []
label_dict = {}
current_label = 0

# Load all images from dataset folder
for folder_name in os.listdir('dataset'):
    folder_path = os.path.join('dataset', folder_name)
    if os.path.isdir(folder_path):
        label_dict[current_label] = folder_name
        print(f"Loading images from: {folder_name}")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces_data.append(img)
                labels.append(current_label)
        current_label += 1

# Train the model
recognizer.train(faces_data, np.array(labels))

# Save model and label mapping
recognizer.save('trainer.yml')
with open('labels.pickle', 'wb') as f:
    pickle.dump(label_dict, f)

print(f"Training complete! Trained {current_label} student(s)")
print("Label mapping:", label_dict)
print("Files saved: trainer.yml and labels.pickle")