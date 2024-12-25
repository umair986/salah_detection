import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm

def process_image(image_path, mp_pose):
    """Process a single image and return the pose landmarks."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(image_rgb)
        
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        row = []
        for landmark in landmarks:
            row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return row
    return None

def create_column_names():
    """Create column names for the DataFrame."""
    columns = []
    for i in range(33):  # MediaPipe tracks 33 landmarks
        columns.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
    return ['class'] + columns

def process_dataset(dataset_dir):
    """Process all images in the dataset directory and create a CSV file."""
    mp_pose = mp.solutions.pose
    all_data = []
    
    # Get all posture folders
    posture_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    
    for posture in posture_folders:
        posture_path = os.path.join(dataset_dir, posture)
        print(f"Processing {posture} poses...")
        
        # Get all images in the posture folder
        image_files = [f for f in os.listdir(posture_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files):
            image_path = os.path.join(posture_path, image_file)
            landmarks = process_image(image_path, mp_pose)
            
            if landmarks:
                # Add class label and landmarks to the data
                row_data = [posture] + landmarks
                all_data.append(row_data)
    
    # Create DataFrame
    columns = create_column_names()
    df = pd.DataFrame(all_data, columns=columns)
    
    # Save to CSV
    output_file = 'pose_landmarks.csv'
    df.to_csv(output_file, index=False)
    print(f"\nProcessing complete! Data saved to {output_file}")
    print(f"Total samples processed: {len(df)}")
    
    return df

# Example usage
if __name__ == "__main__":
   dataset_directory = "C:/Users/mohum/OneDrive/Desktop/ML/salah_detection/dataset_directory" # Replace with your dataset directory path
df = process_dataset(dataset_directory)