import tensorflow as tf
import numpy as np
import cv2
from collections import deque

# Load the MoveNet model
interpreter = tf.lite.Interpreter(model_path='4.tflite')
interpreter.allocate_tensors()

# Define edges for visualization
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

# Confidence threshold and smoothing window
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5
keypoints_buffer = deque(maxlen=SMOOTHING_WINDOW)

# Posture sequence and validation
POSTURE_SEQUENCE = ["Qiyaam", "Rukoo", "Sujood", "Qa'dah", "Salaam"]
current_index = 0

def preprocess_frame(frame):
    """Preprocess the input frame for the model."""
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 256, 256)  # Resize to 256x256
    return tf.cast(img, dtype=tf.uint8)  # Ensure dtype matches model input requirement

def draw_keypoints(frame, keypoints, confidence_threshold):
    """Draw keypoints on the frame."""
    y, x, _ = frame.shape
    keypoints = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    """Draw edges between keypoints on the frame."""
    y, x, _ = frame.shape
    keypoints = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def classify_posture(keypoints):
    """Classify the posture based on keypoints."""
    keypoints = np.squeeze(keypoints)

    # Extract relevant keypoints
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    # Qiyaam (Standing straight with arms folded)
    if (
        nose[2] > CONFIDENCE_THRESHOLD and left_shoulder[2] > CONFIDENCE_THRESHOLD and right_shoulder[2] > CONFIDENCE_THRESHOLD and
        left_hip[2] > CONFIDENCE_THRESHOLD and right_hip[2] > CONFIDENCE_THRESHOLD and
        left_knee[2] > CONFIDENCE_THRESHOLD and right_knee[2] > CONFIDENCE_THRESHOLD
    ):
        if left_elbow[1] < left_shoulder[1] and right_elbow[1] > right_shoulder[1]:
            return "Qiyaam"

    # Rukoo (Bowing with back parallel to the ground)
    if (
        left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1] and
        abs(left_shoulder[0] - left_hip[0]) < 0.1
    ):
        back_angle = calculate_angle(left_hip[:2], left_shoulder[:2], left_knee[:2])
        if 80 <= back_angle <= 100:
            return "Rukoo"

    # Sujood (Prostration)
    if (
        left_knee[2] > CONFIDENCE_THRESHOLD and right_knee[2] > CONFIDENCE_THRESHOLD and
        left_ankle[2] > CONFIDENCE_THRESHOLD and right_ankle[2] > CONFIDENCE_THRESHOLD and
        left_elbow[2] > CONFIDENCE_THRESHOLD and right_elbow[2] > CONFIDENCE_THRESHOLD
    ):
        return "Sujood"

    # Qa'dah (Sitting on knees with straight back)
    if (
        left_knee[2] > CONFIDENCE_THRESHOLD and right_knee[2] > CONFIDENCE_THRESHOLD and
        left_hip[2] > CONFIDENCE_THRESHOLD and right_hip[2] > CONFIDENCE_THRESHOLD and
        abs(left_shoulder[0] - left_hip[0]) > 0.1
    ):
        return "Qa'dah"

    # Salaam (Turning head towards shoulders)
    if nose[2] > CONFIDENCE_THRESHOLD and (
        abs(nose[1] - right_shoulder[1]) < 0.1 or abs(nose[1] - left_shoulder[1]) < 0.1
    ):
        return "Salaam"

    return "Unknown Posture"

def smooth_keypoints(new_keypoints):
    """Smooth keypoints using a rolling average."""
    keypoints_buffer.append(new_keypoints)
    return np.mean(keypoints_buffer, axis=0)

def validate_posture_sequence(detected_posture):
    """Validate the sequence of postures."""
    global current_index
    if detected_posture == POSTURE_SEQUENCE[current_index]:
        current_index += 1
        if current_index == len(POSTURE_SEQUENCE):
            print("Namaz completed properly!")
            current_index = 0  # Reset for the next cycle
    elif detected_posture != "Unknown Posture":
        print("Invalid posture sequence detected.")

# Main loop for video processing
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_image = preprocess_frame(frame)

    # Model inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Smooth keypoints and classify posture
    smoothed_keypoints = smooth_keypoints(keypoints_with_scores)
    posture = classify_posture(smoothed_keypoints)
    validate_posture_sequence(posture)

    # Draw keypoints and connections
    draw_connections(frame, smoothed_keypoints, EDGES, CONFIDENCE_THRESHOLD)
    draw_keypoints(frame, smoothed_keypoints, CONFIDENCE_THRESHOLD)

    # Display posture on the frame
    cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Namaz Posture Detector', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()