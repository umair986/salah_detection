import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Holistic Setup
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Landmark Mapping
LANDMARK_MAPPING = {
    "NOSE": mp_pose.PoseLandmark.NOSE,
    "LEFT_EAR": mp_pose.PoseLandmark.LEFT_EAR,
    "RIGHT_EAR": mp_pose.PoseLandmark.RIGHT_EAR,
    "LEFT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "LEFT_ELBOW": mp_pose.PoseLandmark.LEFT_ELBOW,
    "RIGHT_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "LEFT_WRIST": mp_pose.PoseLandmark.LEFT_WRIST,
    "RIGHT_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST,
    "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
    "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
    "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE,
    "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE,
    "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE,
    "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE
}

# Updated Posture Definitions with More Comprehensive Detection
postures = {
    "Niyyah": {
        "HEAD": {"threshold": (0.4, 0.6), "weight": 1.0},
        "NOSE": {"threshold": (0.4, 0.6), "weight": 1.0}
    },
    "Takbir": {
        "LEFT_SHOULDER": {"threshold": (0.3, 0.7), "weight": 1.0},
        "RIGHT_SHOULDER": {"threshold": (0.3, 0.7), "weight": 1.0},
        "LEFT_ELBOW": {"threshold": (0.4, 0.6), "weight": 1.0},
        "RIGHT_ELBOW": {"threshold": (0.4, 0.6), "weight": 1.0}
    },
    "Qiyam": {
        "LEFT_SHOULDER": {"threshold": (0.40, 0.60), "weight": 1.0},
        "RIGHT_SHOULDER": {"threshold": (0.40, 0.60), "weight": 1.0},
        "LEFT_HIP": {"threshold": (0.50, 0.70), "weight": 1.0},
        "RIGHT_HIP": {"threshold": (0.50, 0.70), "weight": 1.0},
        "LEFT_ANKLE": {"threshold": (0.50, 0.70), "weight": 1.0},
        "RIGHT_ANKLE": {"threshold": (0.50, 0.70), "weight": 1.0}
    },
    "Ruku": {
        "LEFT_SHOULDER": {"threshold": (0.20, 0.40), "weight": 1.2},  
        "RIGHT_SHOULDER": {"threshold": (0.20, 0.40), "weight": 1.2},
        "LEFT_HIP": {"threshold": (0.30, 0.50), "weight": 1.0},
        "RIGHT_HIP": {"threshold": (0.30, 0.50), "weight": 1.0},
        "LEFT_KNEE": {"threshold": (0.60, 0.80), "weight": 1.0},
        "RIGHT_KNEE": {"threshold": (0.60, 0.80), "weight": 1.0}
    },
    "I'tidal": {
        "LEFT_SHOULDER": {"threshold": (0.40, 0.60), "weight": 1.0},
        "RIGHT_SHOULDER": {"threshold": (0.40, 0.60), "weight": 1.0},
        "LEFT_HIP": {"threshold": (0.50, 0.70), "weight": 1.0},
        "RIGHT_HIP": {"threshold": (0.50, 0.70), "weight": 1.0}
    },
    "Sujud": {
        "NOSE": {"threshold": (0.10, 0.30), "weight": 1.5},
        "LEFT_WRIST": {"threshold": (0.10, 0.30), "weight": 1.0},
        "RIGHT_WRIST": {"threshold": (0.10, 0.30), "weight": 1.0},
        "LEFT_ANKLE": {"threshold": (0.10, 0.30), "weight": 1.0},
        "RIGHT_ANKLE": {"threshold": (0.10, 0.30), "weight": 1.0}
    },
    "Jalsa": {
        "LEFT_KNEE": {"threshold": (0.30, 0.50), "weight": 1.0},
        "RIGHT_KNEE": {"threshold": (0.30, 0.50), "weight": 1.0},
        "LEFT_HIP": {"threshold": (0.40, 0.60), "weight": 1.0},
        "RIGHT_HIP": {"threshold": (0.40, 0.60), "weight": 1.0}
    },
    "Tashahhud": {
        "LEFT_KNEE": {"threshold": (0.30, 0.50), "weight": 1.0},
        "RIGHT_KNEE": {"threshold": (0.30, 0.50), "weight": 1.0},
        "LEFT_HIP": {"threshold": (0.40, 0.60), "weight": 1.0},
        "RIGHT_HIP": {"threshold": (0.40, 0.60), "weight": 1.0}
    },
    "Salaam": {
        "LEFT_EAR": {"threshold": (0.40, 0.60), "weight": 1.0},
        "RIGHT_EAR": {"threshold": (0.40, 0.60), "weight": 1.0}
    }
}

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def classify_posture(results):
    """
    Classify the namaz posture based on body landmarks
    """
    if not results.pose_landmarks:
        return "Unknown", {}

    posture_confidences = {}
    landmarks = results.pose_landmarks.landmark
    
    for posture, key_points in postures.items():
        confidence = 0
        total_weight = 0
        
        for point, params in key_points.items():
            # Get the landmark index from MediaPipe
            landmark_index = LANDMARK_MAPPING.get(point)
            if landmark_index is None:
                continue
            
            landmark = landmarks[landmark_index]
            
            # Modify confidence calculation to be more robust
            landmark_conf = landmark.visibility
            if params["threshold"][0] <= landmark_conf <= params["threshold"][1]:
                confidence += params.get("weight", 1.0)
            total_weight += params.get("weight", 1.0)
        
        posture_confidences[posture] = confidence / total_weight if total_weight > 0 else 0
    
    # Determine the most likely posture
    most_likely_posture = max(posture_confidences, key=posture_confidences.get)
    return most_likely_posture, posture_confidences

def main():
    cap = cv2.VideoCapture(0)  
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False
        results = holistic.process(image)
        
        # Draw the pose annotation on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS
        )
        
        # Classify and display the posture
        if results.pose_landmarks:
            posture, confidences = classify_posture(results)
            
            # Display posture and confidence
            display_text = f"Posture: {posture}"
            cv2.putText(
                image, 
                display_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Print detailed confidences (optional)
            print(f"Posture Confidences: {confidences}")
        
        # Display the image
        cv2.imshow('Namaz Posture Detection', cv2.flip(image, 1))
        
        # Exit condition
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break
    
    # Cleanup
    holistic.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()