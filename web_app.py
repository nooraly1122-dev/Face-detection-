from flask import Flask, render_template, jsonify, request, send_file, Response
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import os
import threading
import io

app = Flask(__name__)

# Global variables
recognizer = None
id_to_name = {}
face_detector = None
marked_today = set()
attendance_file = 'attendance.csv'
camera_active = False
camera_capture = None

def initialize_models():
    """Initialize face recognition models"""
    global recognizer, id_to_name, face_detector
    
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer.yml')
        
        with open('labels.pickle', 'rb') as f:
            label_dict = pickle.load(f)
            id_to_name = {v: k for k, v in label_dict.items()}
        
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def mark_attendance(name):
    """Mark attendance for a student"""
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    
    if name in marked_today:
        return False
    
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    else:
        df = pd.read_csv(attendance_file)
    
    if not df[(df['Name'] == name) & (df['Date'] == today)].empty:
        marked_today.add(name)
        return False
    
    new_entry = pd.DataFrame([[name, today, now]], columns=['Name', 'Date', 'Time'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    marked_today.add(name)
    return True

def gen_frames():
    """Generate frames from camera"""
    global camera_active, camera_capture
    
    camera_capture = cv2.VideoCapture('http://localhost:4747')
    
    if not camera_capture.isOpened():
        print("Trying WiFi...")
        camera_capture = cv2.VideoCapture('http://192.168.100.97:4747')
    
    if not camera_capture.isOpened():
        print("Using default camera...")
        camera_capture = cv2.VideoCapture(0)
    
    camera_active = True
    
    while camera_active:
        ret, frame = camera_capture.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            try:
                id_pred, conf = recognizer.predict(roi_gray)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if conf < 90:
                    name = id_to_name.get(id_pred, "Unknown")
                    
                    if mark_attendance(name):
                        cv2.putText(frame, f"MARKED: {name}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"{name} (Marked)", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "UNKNOWN", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except:
                cv2.putText(frame, "UNKNOWN", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Streaming video feed"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    """Stop camera stream"""
    global camera_active, camera_capture
    camera_active = False
    if camera_capture:
        camera_capture.release()
    return jsonify({'status': 'Camera stopped'})

@app.route('/get_attendance')
def get_attendance():
    """Get attendance records"""
    try:
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            return jsonify(df.to_dict('records'))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_today_attendance')
def get_today_attendance():
    """Get today's attendance only"""
    try:
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            today = datetime.now().strftime("%Y-%m-%d")
            today_df = df[df['Date'] == today]
            return jsonify(today_df.to_dict('records'))
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_attendance')
def download_attendance():
    """Download attendance CSV"""
    try:
        if os.path.exists(attendance_file):
            return send_file(attendance_file, as_attachment=True, download_name=f'attendance_{datetime.now().strftime("%Y-%m-%d")}.csv')
        return jsonify({'error': 'No attendance file found'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/add_student', methods=['POST'])
def add_student():
    """Add new student (placeholder)"""
    data = request.json
    return jsonify({'status': 'success', 'message': 'Run train_model.py to add new student'})

@app.route('/get_students')
def get_students():
    """Get list of registered students"""
    students = list(id_to_name.values())
    return jsonify(students)

@app.route('/stats')
def get_stats():
    """Get attendance statistics"""
    try:
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = len(df[df['Date'] == today])
            total_students = len(id_to_name)
            total_attendance = len(df)
            
            return jsonify({
                'today_marked': today_count,
                'total_students': total_students,
                'total_attendance_records': total_attendance
            })
        return jsonify({
            'today_marked': 0,
            'total_students': len(id_to_name),
            'total_attendance_records': 0
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Initializing models...")
    if initialize_models():
        print("✓ Models loaded successfully!")
        print("Starting web dashboard on http://localhost:5000")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("✗ Failed to load models. Make sure trainer.yml and labels.pickle exist.")
