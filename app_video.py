import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import winsound
from collections import deque

# Load Haar Cascade for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the trained model
model = load_model("drowsiness_detection_model_v2.h5")

# Parameters for stability
drowsy_threshold = 8  # Number of consecutive frames with closed eyes to consider drowsy
frame_window = 7  # Number of frames to use for averaging predictions
predictions = deque(maxlen=frame_window)  # Store last N predictions

def get_eye_region(frame):
    # Convert to grayscale for face/eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 4)
        for (ex, ey, ew, eh) in eyes:
            eye_region = face_roi[ey:ey+eh, ex:ex+ew]
            return cv2.resize(eye_region, (64, 64))  # Resize to model's input size
    
    return None

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(1)

closed_eye_frames = 0  # Counter for consecutive closed-eye frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the eye region from the frame
    eye_region = get_eye_region(frame)
    if eye_region is not None:
        # Process eye region for prediction
        img_array = img_to_array(eye_region) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        # Check if eyes are closed
        if prediction[0][0] < 0.5:
            predictions.append(0)  # 0 for closed eyes
        else:
            predictions.append(1)  # 1 for open eyes

        # Calculate average over frame window
        avg_prediction = sum(predictions) / len(predictions)
        if avg_prediction < 0.5:  # Majority closed eyes
            status = "Drowsy (Eyes Closed)"
            closed_eye_frames += 1
        else:
            status = "Active (Eyes Open)"
            closed_eye_frames = 0

        # Trigger alert if drowsy threshold reached
        if closed_eye_frames >= drowsy_threshold:
            alert = "ALERT: Drowsy"
            color = (0, 0, 255)
            winsound.Beep(1000, 500)  # Beep for alert
        else:
            alert = "Active"
            color = (0, 255, 0)

        # Display results
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, alert, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        status = "Person not in frame"
        alert = "ALERT : Face not detected"
        color = (0, 0, 255)
        winsound.Beep(1000, 500)
        # Display results
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, alert, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
