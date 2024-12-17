import tensorflow as tf
import numpy as np
import cv2

# Load the MoveNet model
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

# Define edges for visualization
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

# Function to preprocess the input frame
def preprocess_frame(frame):
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    return tf.cast(img, dtype=tf.float32)

# Draw keypoints on the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, _ = frame.shape
    keypoints = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Draw edges on the frame
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, _ = frame.shape
    keypoints = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]

        if c1 > confidence_threshold and c2 > confidence_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# Function to classify postures
def classify_posture(keypoints):
    keypoints = np.squeeze(keypoints)
    
    # Extract relevant keypoints
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    # Qiyaam (Standing straight)
    if (
        nose[2] > 0.5 and left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5 and
        left_hip[2] > 0.5 and right_hip[2] > 0.5 and
        left_knee[2] > 0.5 and right_knee[2] > 0.5
    ):
        if left_elbow[1] < left_shoulder[1] and right_elbow[1] > right_shoulder[1]:
            return "Qiyaam"

    # Rukoo (Bowing)
    if (
        left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1] and
        abs(left_shoulder[0] - left_hip[0]) < 0.1
    ):
        return "Rukoo"

    # Sujood (Prostration)
    if (
        left_knee[2] > 0.5 and right_knee[2] > 0.5 and
        left_ankle[2] > 0.5 and right_ankle[2] > 0.5 and
        left_wrist[2] > 0.5 and right_wrist[2] > 0.5
    ):
        return "Sujood"

    # Qa'dah (Sitting)
    if (
        left_knee[2] > 0.5 and right_knee[2] > 0.5 and
        left_hip[2] > 0.5 and right_hip[2] > 0.5 and
        abs(left_shoulder[0] - left_hip[0]) > 0.1
    ):
        return "Qa'dah"

    # Salaam (Turning head to shoulders)
    if nose[2] > 0.5 and (
        abs(nose[1] - right_shoulder[1]) < 0.1 or abs(nose[1] - left_shoulder[1]) < 0.1
    ):
        return "Salaam"

    return "Unknown Posture"

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

    # Classify the posture
    posture = classify_posture(keypoints_with_scores)
    print(f"Detected Posture: {posture}")

    # Draw keypoints and connections
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    # Display posture on the frame
    cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Namaz Posture Detector', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()