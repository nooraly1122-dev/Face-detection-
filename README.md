# Facial Recognition Attendance System

An automated, local-network attendance management system leveraging Computer Vision to track and log student presence in real-time.

## Key Features
- Real-time Detection and Extraction: Powered by OpenCV Haar Cascades for high-speed face tracking.
- LBPH Recognition Engine: Utilizes Local Binary Patterns Histograms for localized pattern classification (Confidence threshold < 90).
- Data Persistence Layer: Auto-manages backend logging dynamically inside attendance.csv using Python Pandas.
- Network Camera Bridge: Configured to ingest wireless IP video pipelines natively (http://localhost:4747).
- Anti-Duplication Guard: Enforces runtime caching tracking (marked_today validation set) to prevent repetitive logs for the same user within a single calendar date.

## Tech Stack and Dependencies
- Language: Python 3.x
- Libraries: opencv-python, numpy, pandas, pickle

## Navigation and System Flow
1. Bootup Phase: System initializes file system dependencies, verifies or creates attendance.csv, and loads pre-compiled weights from trainer.yml and labels.pickle.
2. Stream Phase: Establishes live frames capture loop via local server stream port.
3. Processing Phase: Converts frames to grayscale, detects spatial matrices via Cascade parameters, and executes LBPH distance mapping queries.
4. Logging Phase: If verified, single-instance verification flags update to the flat file.
5. Teardown Phase: Pressing 'q' terminates process memory loops, closes operational window rendering, and dumps an active pandas preview terminal summary of the day's attendance log.
