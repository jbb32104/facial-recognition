import cv2
import face_recognition
import os
from gpiozero import LED
from time import sleep

# Initialize GPIO pin
face_detected_pin = LED(17)  # GPIO17 (physical pin 11)

# Load known face encodings
def load_known_faces(known_faces_dir="known_faces"):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(known_faces_dir):
        person_path = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_path):
            for filename in os.listdir(person_path):
                img_path = os.path.join(person_path, filename)
                if os.path.isfile(img_path):
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name.replace("_", " "))
                    else:
                        print(f"No face found in {img_path}")
    return known_encodings, known_names


known_faces_dir = "known_faces"
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

known_encodings, known_names = load_known_faces(known_faces_dir)

print(f"Loaded {len(known_encodings)} face(s) for {len(set(known_names))} person(s).")
print("Press 'q' to quit the application.")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error accessing the camera.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Detect faces
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    face_detected = False  # Track if any known face is detected

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            face_detected = True

        # Scale face location back to original frame size
        top, right, bottom, left = [x * 4 for x in face_location]

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Output high voltage if a known face is detected
    if face_detected:
        face_detected_pin.on()
    else:
        face_detected_pin.off()

    # Display the video feed
    cv2.imshow("Video", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
face_detected_pin.off()
video_capture.release()
cv2.destroyAllWindows()
