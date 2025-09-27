import cv2
import face_recognition
import numpy as np
import os
import threading
from datetime import datetime, timedelta
from pymongo import MongoClient

# ---------------------------
# Step 1: MongoDB Setup
# ---------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_system"]
students_collection = db["students"]        # Stores student info
attendance_collection = db["attendance"]    # Stores attendance records
unknown_collection = db["unknown_faces"]    # Stores unknown faces

# ---------------------------
# Step 2: Load known faces
# ---------------------------
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "known_faces_encodings.npz"
UNKNOWN_FACES_DIR = "unknown_faces"
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

known_encodings = []
known_names = []

if os.path.exists(ENCODINGS_FILE):
    data = np.load(ENCODINGS_FILE, allow_pickle=True)
    known_encodings = data["encodings"].tolist()
    known_names = data["names"].tolist()
    print(f"[INFO] Loaded {len(known_encodings)} known faces from cache")
else:
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"[WARN] No face found in {filename}")
            else:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
                print(f"[INFO] Added {filename} -> {name}")

                # Optional: Insert student info into MongoDB if not exists
                if students_collection.count_documents({"name": name}) == 0:
                    students_collection.insert_one({
                        "name": name,
                        "roll_no": "",    # Add manually later
                        "standard": ""    # Add manually later
                    })

    np.savez(ENCODINGS_FILE, encodings=known_encodings, names=known_names)
    print(f"[INFO] Processed and saved {len(known_encodings)} known faces")

# ---------------------------
# Step 3: Attendance Tracker
# ---------------------------
attendance_set = set()  # prevent duplicate attendance per session

def mark_attendance(name):
    """Mark attendance in MongoDB."""
    if name not in attendance_set:
        now = datetime.now()
        attendance_record = {
            "name": name,
            "timestamp": now
        }
        attendance_collection.insert_one(attendance_record)
        attendance_set.add(name)
        print(f"[ATTENDANCE] {name} marked at {now}")

# ---------------------------
# Step 4: Threaded Video Stream
# ---------------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

vs = VideoStream(0)

# ---------------------------
# Step 5: Real-time Face Recognition
# ---------------------------
unknown_id = 0
recent_unknowns = []
UNKNOWN_TOLERANCE = 0.6
UNKNOWN_THROTTLE_SECONDS = 10
MAX_RECENT = 50

while True:
    frame = vs.read()
    if frame is None:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    face_names = []
    recognized_count = 0

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                recognized_count += 1
                mark_attendance(name)
            else:
                # Unknown face detection with throttling
                is_new_unknown = True
                now = datetime.now()
                for unk in recent_unknowns:
                    distance = np.linalg.norm(unk["encoding"] - face_encoding)
                    seconds_since_seen = (now - unk["last_seen"]).total_seconds()
                    if distance < UNKNOWN_TOLERANCE and seconds_since_seen < UNKNOWN_THROTTLE_SECONDS:
                        is_new_unknown = False
                        break

                if is_new_unknown:
                    # Save unknown face
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    face_crop = frame[top:bottom, left:right]
                    if face_crop.size > 0:
                        filename = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{unknown_id}.jpg")
                        cv2.imwrite(filename, face_crop)

                        # Store in MongoDB
                        unknown_collection.insert_one({
                            "image_path": filename,
                            "timestamp": now
                        })

                        print(f"[ALERT] Unknown face saved: {filename}")
                        unknown_id += 1

                        recent_unknowns.append({"encoding": face_encoding, "last_seen": now})
                        if len(recent_unknowns) > MAX_RECENT:
                            recent_unknowns.pop(0)

        face_names.append(name)

    # Draw boxes and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Recognized Faces: {recognized_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("MongoDB Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()
