import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
    Deepfake Detector with RGB as input layer, D as output layer
"""


# Load image
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = transforms.ToTensor()(img)  # Convert to PyTorch tensor
    return img.unsqueeze(0)  # Add batch dimension


# Load depth map
def load_depth_map(depth_map_path):
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is not None:
        depth_map = cv2.resize(depth_map, (224, 224))
        depth_map = depth_map.astype(np.float32) / 255.0  # Normalize to [0, 1]
        depth_map = transforms.ToTensor()(depth_map)  # Convert to PyTorch tensor
        return depth_map.unsqueeze(0)  # Add batch dimension


# Load images, depth maps and assign label
def load_data(data_dir):
    images = []
    depth_maps = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(data_dir, filename)
            data_depth_dir = data_dir.replace('faces', 'depth_maps')
            depth_map_path = os.path.join(data_depth_dir, filename + '_depth.png')
            image = load_image(image_path)
            depth_map = load_depth_map(depth_map_path)
            label = 0 if 'original' in data_dir else 1
            images.append(image)
            depth_maps.append(depth_map)
            labels.append(label)
            print(f'Image loaded {image_path} and {depth_map_path} as {label}')
    return torch.cat(images), torch.cat(depth_maps), torch.tensor(labels)


# Deepfake Detection Model
class DeepfakeDetector(nn.Module):
    """
        Deepfake Detector Class (RGB Input, D Output)

        :returns: deepfake classification
    """
    def __init__(self):
        super(DeepfakeDetector, self).__init__()

        # MobileNet for RGB images
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Small convolutional network for depth maps
        self.depth_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully connected layers
        self.fc1 = nn.Linear(1280 + 32, 256)  # Concatenate features from MobileNet and depth network
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, depth_maps):
        # Process RGB images with MobileNet
        rgb_features = self.mobilenet.features(images)
        rgb_features = self.mobilenet_avgpool(rgb_features)
        rgb_features = torch.flatten(rgb_features, 1)

        # Process depth maps with the small convolutional network
        depth_features = self.depth_net(depth_maps)
        depth_features = torch.flatten(depth_features, 1)

        # Concatenate features and pass through fully connected layers
        fused_features = torch.cat((rgb_features, depth_features), dim=1)
        x = nn.functional.relu(self.fc1(fused_features))
        x = self.sigmoid(self.fc2(x))
        return x


# Training Function
def train_model(epochs=15):
    # Load data
    original_images, original_depth_maps, original_labels = load_data('faceforensics_faces/original_sequences/youtube/c23/videos')
    deepfake_images, deepfake_depth_maps, deepfake_labels = load_data('faceforensics_faces/manipulated_sequences/Deepfakes/c23/videos')

    # Combine data
    images = torch.cat((original_images, deepfake_images))
    depth_maps = torch.cat((original_depth_maps, deepfake_depth_maps))
    labels = torch.cat((original_labels, deepfake_labels))

    # Split data into training and test sets
    images_train, images_test, depth_maps_train, depth_maps_test, labels_train, labels_test = train_test_split(
        images, depth_maps, labels, test_size=0.2, random_state=42
    )

    # Further split the test set into validation and test sets
    images_val, images_test, depth_maps_val, depth_maps_test, labels_val, labels_test = train_test_split(
        images_test, depth_maps_test, labels_test, test_size=0.5, random_state=42
    )

    # Create data loaders
    train_loader = DataLoader(list(zip(images_train, depth_maps_train, labels_train)), batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(list(zip(images_val, depth_maps_val, labels_val)), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(list(zip(images_test, depth_maps_test, labels_test)), batch_size=32, shuffle=False, num_workers=4)

    # Create the model
    model = DeepfakeDetector()
    device = torch.device("mps") if torch.mps else torch.device("cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Crossentropy
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)  # Adamax optimizer

    # Train the model
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        print(f"Training epoch {epoch+1}/{num_epochs} on {device}")

        # Get the total number of batches
        total_batches = len(train_loader)
        for batch, (images, depth_maps, labels) in enumerate(train_loader):
            images = images.to(device)
            depth_maps = depth_maps.to(device)
            labels = labels.to(device).float()

            # Normalize RGB images (3 channels)
            rgb_mean = [0.485, 0.456, 0.406]
            rgb_std = [0.229, 0.224, 0.225]
            for i in range(3):
                images[:, i, :, :] = (images[:, i, :, :] - rgb_mean[i]) / rgb_std[i]

            # Normalize depth maps (1 channel)
            depth_maps = (depth_maps - 0.5) / 0.5

            optimizer.zero_grad()
            outputs = model(images, depth_maps)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate and print progress
            progress = (batch + 1) / total_batches * 100
            print(f"Progress: {progress:.2f}%")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            for images, depth_maps, labels in val_loader:
                images = images.to(device)
                depth_maps = depth_maps.to(device)
                labels = labels.to(device).float()

                # Normalize inputs as in training
                for i in range(3):
                    images[:, i, :, :] = (images[:, i, :, :] - rgb_mean[i]) / rgb_std[i]
                depth_maps = (depth_maps - 0.5) / 0.5

                outputs = model(images, depth_maps)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                val_acc += (preds == labels).sum().item() / len(labels)

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'deepfake_detector_output_d.pth')

    # Evaluation
    evaluate_model(test_loader)


# Evaluation Function
def evaluate_model(test_loader=None):
    # Load the saved model
    model = DeepfakeDetector()
    model.load_state_dict(torch.load('deepfake_detector_output_d.pth'))
    model.eval()

    if test_loader is None:
        # Load data
        original_images, original_depth_maps, original_labels = load_data('faceforensics_faces/original_sequences/youtube/c23/videos')
        deepfake_images, deepfake_depth_maps, deepfake_labels = load_data('faceforensics_faces/manipulated_sequences/Deepfakes/c23/videos')

        images = torch.cat((original_images, deepfake_images))
        depth_maps = torch.cat((original_depth_maps, deepfake_depth_maps))
        labels = torch.cat((original_labels, deepfake_labels))

        # Split data into training and test sets
        images_train, images_test, depth_maps_train, depth_maps_test, labels_train, labels_test = train_test_split(
            images, depth_maps, labels, test_size=0.2, random_state=42
        )

        images_val, images_test, depth_maps_val, depth_maps_test, labels_val, labels_test = train_test_split(
            images_test, depth_maps_test, labels_test, test_size=0.5, random_state=42
        )

        test_loader = DataLoader(list(zip(images_test, depth_maps_test, labels_test)), batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("mps") if torch.mps else torch.device("cpu")
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, depth_maps, labels in test_loader:
            images = images.to(device)
            depth_maps = depth_maps.to(device)
            labels = labels.to(device).float()

            # Normalize inputs as in training
            rgb_mean = [0.485, 0.456, 0.406]
            rgb_std = [0.229, 0.224, 0.225]
            for i in range(3):
                images[:, i, :, :] = (images[:, i, :, :] - rgb_mean[i]) / rgb_std[i]
            depth_maps = (depth_maps - 0.5) / 0.5

            outputs = model(images, depth_maps)
            outputs = outputs.squeeze(1)
            preds = (outputs > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", confusion_mat)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Original", "Deepfake"],
                yticklabels=["Original", "Deepfake"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Output Depth')
    textstr = '\n'.join((
        f'Accuracy: {accuracy * 100:.2f}%',
        f'Precision: {precision * 100:.2f}%',
        f'Recall: {recall * 100:.2f}%',
        f'F1-Score: {f1 * 100:.2f}%'))
    plt.gcf().text(0.98, 0.02, textstr, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    plt.savefig('plots/confusion_matrix_output_d.png', dpi=300)
    plt.show()

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.fill_between(fpr, tpr, color='orange', alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Output Depth')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve_output_d.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    train_model()
    #evaluate_model()
