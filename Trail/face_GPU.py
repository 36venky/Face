import cv2
import face_recognition
import numpy as np
import os
import threading
import faiss  # for fast similarity search

# ---------------------------
# Step 1: Load known faces with FAISS index
# ---------------------------
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "known_faces_encodings.npz"

known_encodings = []
known_names = []

if os.path.exists(ENCODINGS_FILE):
    data = np.load(ENCODINGS_FILE, allow_pickle=True)
    known_encodings = np.array(data["encodings"].tolist()).astype("float32")
    known_names = data["names"].tolist()
    print(f"[INFO] Loaded {len(known_encodings)} known faces from cache")
else:
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

    known_encodings = np.array(known_encodings).astype("float32")
    np.savez(ENCODINGS_FILE, encodings=known_encodings, names=known_names)
    print(f"[INFO] Processed and saved {len(known_encodings)} known faces")

# Build FAISS index for fast search
index = faiss.IndexFlatL2(128)  # 128-d embeddings
index.add(known_encodings)

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
# Step 3: Real-time recognition with optimizations
# ---------------------------
process_frame = True  # Skip every 2nd frame for speed
MODEL_TYPE = "cnn" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "hog"
print(f"[INFO] Using model: {MODEL_TYPE}")

while True:
    frame = vs.read()
    if frame is None:
        continue

    # Only process every other frame
    if process_frame:
        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small, model=MODEL_TYPE)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        recognized_count = 0

        for encoding in face_encodings:
            encoding = np.array([encoding]).astype("float32")

            # Search in FAISS index
            D, I = index.search(encoding, 1)  # distance, index
            if D[0][0] < 0.6:  # threshold
                name = known_names[I[0][0]]
                recognized_count += 1
            else:
                name = "Unknown"
            face_names.append(name)

    process_frame = not process_frame

    # Draw results
    if face_locations:
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    # Show recognized face count
    cv2.putText(frame, f"Recognized Faces: {recognized_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vs.stop()
cv2.destroyAllWindows()