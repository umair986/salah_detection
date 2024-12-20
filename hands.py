import numpy as np
import cv2
import mediapipe as mp
from enum import Enum
import json
from datetime import datetime
from collections import deque

class PrayerPose(Enum):
    NIYYAH = "Niyyah"
    TAKBIR = "Takbir"
    QIYAM = "Qiyam"
    RUKU = "Ruku"
    ITIDAL = "Itidal"
    SUJUD = "Sujud"
    JALSA = "Jalsa"
    TASHAHHUD = "Tashahhud"
    SALAM_RIGHT = "Salam Right"
    SALAM_LEFT = "Salam Left"
    UNKNOWN = "Unknown"

class NamazPoseDetector:
    def _init_(self, confidence_threshold=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        self.confidence_threshold = confidence_threshold
        self.pose_history = deque(maxlen=30)
        self.current_sequence = []
        self.debug_data = []
        self.debug_info = {}

    def get_3d_landmarks(self, results):
        """Extract 3D landmarks with confidence filtering."""
        if not results.pose_world_landmarks:
            return None

        landmarks_3d = []
        for landmark in results.pose_world_landmarks.landmark:
            landmarks_3d.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        return landmarks_3d

    def classify_pose(self, landmarks_3d):
        """Classify the prayer pose based on 3D landmarks."""
        if not landmarks_3d:
            return PrayerPose.UNKNOWN

        # Reset debug info
        self.debug_info = {}

        # Check each pose with debug information
        if self._is_qiyam_position(landmarks_3d):
            return PrayerPose.QIYAM
        elif self._is_ruku_position(landmarks_3d):
            return PrayerPose.RUKU
        elif self._is_sujud_position(landmarks_3d):
            return PrayerPose.SUJUD
        elif self._is_jalsa_position(landmarks_3d):
            return PrayerPose.JALSA
        elif self._is_tashahhud_position(landmarks_3d):
            return PrayerPose.TASHAHHUD
        elif self._is_niyyah_position(landmarks_3d):
            return PrayerPose.NIYYAH
        elif self._is_takbir_position(landmarks_3d):
            return PrayerPose.TAKBIR

        self.debug_info['pose_classification'] = 'No matching pose found'
        return PrayerPose.UNKNOWN

    def _is_qiyam_position(self, landmarks_3d):
        """Check if pose matches Qiyam position (standing with hands folded)."""
        nose = landmarks_3d[0]
        left_shoulder = landmarks_3d[11]
        right_shoulder = landmarks_3d[12]
        left_wrist = landmarks_3d[15]
        right_wrist = landmarks_3d[16]
        left_ankle = landmarks_3d[27]
        right_ankle = landmarks_3d[28]

        vertical_aligned = abs(left_shoulder['y'] - right_shoulder['y']) < 0.1
        
        hands_folded = (
            abs(left_wrist['y'] - right_wrist['y']) < 0.15 and
            abs(left_wrist['x'] - right_wrist['x']) < 0.2 and
            left_wrist['y'] < left_shoulder['y'] and
            right_wrist['y'] < right_shoulder['y']
        )

        standing = (
            nose['y'] < left_shoulder['y'] < left_ankle['y'] and
            nose['y'] < right_shoulder['y'] < right_ankle['y']
        )

        self.debug_info['qiyam'] = {
            'vertical_aligned': vertical_aligned,
            'hands_folded': hands_folded,
            'standing': standing
        }

        return vertical_aligned and hands_folded and standing

    def _is_ruku_position(self, landmarks_3d):
        """Check if pose matches Ruku position (bowing)."""
        nose = landmarks_3d[0]
        left_shoulder = landmarks_3d[11]
        right_shoulder = landmarks_3d[12]
        left_hip = landmarks_3d[23]
        right_hip = landmarks_3d[24]
        left_ankle = landmarks_3d[27]
        right_ankle = landmarks_3d[28]

        spine_angle = abs(90 - self._calculate_angle(
            [left_shoulder['x'], left_shoulder['y']],
            [left_hip['x'], left_hip['y']],
            [left_hip['x'] + 1, left_hip['y']]
        ))

        bending_forward = (
            nose['y'] > left_shoulder['y'] and
            nose['y'] > right_shoulder['y'] and
            left_shoulder['y'] > left_hip['y'] and
            right_shoulder['y'] > right_hip['y']
        )

        knees_straight = (
            left_hip['y'] < left_ankle['y'] and
            right_hip['y'] < right_ankle['y']
        )

        self.debug_info['ruku'] = {
            'spine_angle': spine_angle,
            'bending_forward': bending_forward,
            'knees_straight': knees_straight
        }

        return spine_angle < 30 and bending_forward and knees_straight

    def _is_sujud_position(self, landmarks_3d):
        """Check if pose matches Sujud position (prostration)."""
        nose = landmarks_3d[0]
        left_knee = landmarks_3d[25]
        right_knee = landmarks_3d[26]
        left_ankle = landmarks_3d[27]
        right_ankle = landmarks_3d[28]

        head_low = (
            nose['y'] > left_knee['y'] and
            nose['y'] > right_knee['y']
        )

        knees_bent = (
            left_knee['y'] > left_ankle['y'] and
            right_knee['y'] > right_ankle['y']
        )

        self.debug_info['sujud'] = {
            'head_low': head_low,
            'knees_bent': knees_bent
        }

        return head_low and knees_bent

    def _is_jalsa_position(self, landmarks_3d):
        """Check if pose matches Jalsa position (sitting between prostrations)."""
        nose = landmarks_3d[0]
        left_hip = landmarks_3d[23]
        right_hip = landmarks_3d[24]
        left_knee = landmarks_3d[25]
        right_knee = landmarks_3d[26]
        left_ankle = landmarks_3d[27]
        right_ankle = landmarks_3d[28]

        sitting = (
            left_hip['y'] > left_knee['y'] - 0.1 and
            right_hip['y'] > right_knee['y'] - 0.1
        )

        back_straight = (
            nose['y'] < left_hip['y'] and
            nose['y'] < right_hip['y']
        )

        feet_position = (
            left_ankle['y'] > left_knee['y'] and
            right_ankle['y'] > right_knee['y']
        )

        self.debug_info['jalsa'] = {
            'sitting': sitting,
            'back_straight': back_straight,
            'feet_position': feet_position
        }

        return sitting and back_straight and feet_position

    def _is_tashahhud_position(self, landmarks_3d):
        """Check if pose matches Tashahhud position (final sitting)."""
        basic_sitting = self._is_jalsa_position(landmarks_3d)
        
        right_wrist = landmarks_3d[16]
        right_thumb = landmarks_3d[22]
        
        finger_raised = (
            right_thumb['y'] < right_wrist['y']
        )

        self.debug_info['tashahhud'] = {
            'basic_sitting': basic_sitting,
            'finger_raised': finger_raised
        }

        return basic_sitting and finger_raised

    def _is_niyyah_position(self, landmarks_3d):
        """Check if pose matches Niyyah position (standing with intention)."""
        nose = landmarks_3d[0]
        left_shoulder = landmarks_3d[11]
        right_shoulder = landmarks_3d[12]
        left_wrist = landmarks_3d[15]
        right_wrist = landmarks_3d[16]
        left_ankle = landmarks_3d[27]
        right_ankle = landmarks_3d[28]

        standing = (
            nose['y'] < left_shoulder['y'] < left_ankle['y'] and
            nose['y'] < right_shoulder['y'] < right_ankle['y']
        )

        hands_by_sides = (
            abs(left_wrist['x'] - left_shoulder['x']) < 0.2 and
            abs(right_wrist['x'] - right_shoulder['x']) < 0.2 and
            left_wrist['y'] > left_shoulder['y'] and
            right_wrist['y'] > right_shoulder['y']
        )

        self.debug_info['niyyah'] = {
            'standing': standing,
            'hands_by_sides': hands_by_sides
        }

        return standing and hands_by_sides

    def _is_takbir_position(self, landmarks_3d):
        """Check if pose matches Takbir position (raising hands to ears)."""
        nose = landmarks_3d[0]
        left_shoulder = landmarks_3d[11]
        right_shoulder = landmarks_3d[12]
        left_wrist = landmarks_3d[15]
        right_wrist = landmarks_3d[16]
        left_ear = landmarks_3d[7]
        right_ear = landmarks_3d[8]

        hands_raised = (
            abs(left_wrist['y'] - left_ear['y']) < 0.15 and
            abs(right_wrist['y'] - right_ear['y']) < 0.15
        )

        back_straight = (
            nose['y'] < left_shoulder['y'] and
            nose['y'] < right_shoulder['y']
        )

        hands_apart = (
            abs(left_wrist['x'] - right_wrist['x']) > 0.3
        )

        self.debug_info['takbir'] = {
            'hands_raised': hands_raised,
            'back_straight': back_straight,
            'hands_apart': hands_apart
        }

        return hands_raised and back_straight and hands_apart

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def process_frame(self, frame):
        """Process a single frame and return pose landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results

def draw_visualization(frame, results, current_pose, debug_info):
    """Draw pose landmarks and debug information on the frame."""
    # Draw pose landmarks
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
    )
    
    # Draw pose name
    cv2.putText(frame, f"Detected Pose: {current_pose.value}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw debug information
    y_pos = 60
    for pose_name, conditions in debug_info.items():
        if isinstance(conditions, dict):  # Check if conditions is a dictionary
            cv2.putText(frame, f"{pose_name}:", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
            for condition, value in conditions.items():
                color = (0, 255, 0) if value else (0, 0, 255)
                cv2.putText(frame, f"  {condition}: {value}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 20
    
    return frame

def main():
    detector = NamazPoseDetector()
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = detector.process_frame(frame)
        
        # Get landmarks and classify pose
        landmarks_3d = detector.get_3d_landmarks(results)
        if landmarks_3d and results.pose_landmarks:
            current_pose = detector.classify_pose(landmarks_3d)
            
            # Draw visualization with debug info
            frame = draw_visualization(frame, results, current_pose, detector.debug_info)
        
        cv2.imshow('Namaz Pose Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    main()