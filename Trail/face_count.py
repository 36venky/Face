import cv2
import face_recognition
import numpy as np
import os
import threading

# ---------------------------
# Step 1: Load known faces
# ---------------------------
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "known_faces_encodings.npz"

if os.path.exists(ENCODINGS_FILE):
    data = np.load(ENCODINGS_FILE, allow_pickle=True)
    known_encodings = data["encodings"].tolist()
    known_names = data["names"].tolist()
    print(f"[INFO] Loaded {len(known_encodings)} known faces from cache")
else:
    known_encodings = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"[WARN] No face found in {filename}")
            else:
                # Take the first (and only) encoding
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

    print(f"[INFO] Loaded {len(known_encodings)} known faces")
    np.savez(ENCODINGS_FILE, encodings=known_encodings, names=known_names)


# ---------------------------
# Step 2: Threaded video stream
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
# Step 3: Real-time face recognition with counting
# ---------------------------
while True:
    frame = vs.read()
    if frame is None:
        continue

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces (cnn is more accurate, hog is faster)
    face_locations = face_recognition.face_locations(rgb_small, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    face_names = []
    recognized_count = 0  # Counter for recognized faces

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                recognized_count += 1  # Increment counter when face is recognized
        face_names.append(name)

    # Draw boxes, labels, and recognized count
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Display recognized face count on the frame
    cv2.putText(frame, f"Recognized Faces: {recognized_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()
