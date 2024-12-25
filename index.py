import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from enum import Enum

class PrayerPose(Enum):
    QIYAM = "Qiyam"
    RUKU = "Ruku"
    SUJUD = "Sujud"
    JALSA = "Jalsa"
    TASHAHHUD = "Tashahhud"
    UNKNOWN = "Unknown"

class NamazPoseDataset(Dataset):
    def __init__(self, root_dir, transform=None):  # Use __init__ instead of _init_
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [pose.value for pose in PrayerPose if pose != PrayerPose.UNKNOWN]
        self.image_paths = []
        self.labels = []
        
        # Load dataset
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found for {class_name}")
                continue
                
            print(f"Loading images for {class_name}...")
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
            
            print(f"Found {len([x for x in self.labels if x == class_idx])} images for {class_name}")

    def __len__(self):  # Use __len__ instead of _len_
        return len(self.image_paths)

    def __getitem__(self, idx):  # Use __getitem__ instead of _getitem_
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class NamazPoseCNN(nn.Module):
    def __init__(self, num_classes):  # Use __init__
        super(NamazPoseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = NamazPoseDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if len(dataset) == 0:
        print("No images found in the dataset! Check your data directory structure.")
        return

    # Create model
    num_classes = len([pose for pose in PrayerPose if pose != PrayerPose.UNKNOWN])
    model = NamazPoseCNN(num_classes).to(device)
    
    # Set up loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    print("Training completed!")
    
    # Save the model
    torch.save(model.state_dict(), 'namaz_pose_model.pth')
    print("Model saved as 'namaz_pose_model.pth'")

if __name__ == "__main__":
    print("Starting the script...")
    dataset_directory = "dataset_directory"  # Update this path
    train_model(dataset_directory)
