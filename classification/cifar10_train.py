import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from svg import Parser, Rasterizer
import torchvision.transforms as transforms
import wandb


# Custom Dataset for CIFAR-10 JPEG images
class CIFAR10JPGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory containing the JPEG images.
            transform (callable, optional): Transform to apply to each image.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List only .jpg files in the directory.
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get filename and extract label from its first digit.
        filename = self.image_files[idx]
        # Assumes filename is a 6-digit number, e.g., "012345.jpg"
        label = int(filename[0])
        image_path = os.path.join(self.root_dir, filename)
        # Open the image and convert to RGB (CIFAR-10 images are colored)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class CIFAR10SVGDataset(Dataset):
    def __init__(self, root_dir, shapes, mode, transform=None):
        """
        Args:
            root_dir (str): Directory containing the JPEG images.
            transform (callable, optional): Transform to apply to each image.
        """
        self.root_dir = os.path.join(root_dir, f'{shapes}_shapes_mode{mode}')
        self.transform = transform
        # List only .svg files in the directory.
        self.image_files = [f for f in os.listdir(self.root_dir) if f.endswith('.svg')]
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get filename and extract label from its first digit.
        filename = self.image_files[idx]
        # Assumes filename is a 6-digit number, e.g., "012345.jpg"
        label = int(filename[0])
        image_path = os.path.join(self.root_dir, filename)
        # Open the image and convert to RGB (CIFAR-10 images are colored)
        with open(image_path, "r", encoding="utf-8") as f:
            svg = f.read()

        svg_data = Parser.parse(svg)

        rast = Rasterizer()
        buff = rast.rasterize(svg_data, svg_data.width, svg_data.height)
        image = Image.frombytes('RGBA', (svg_data.width, svg_data.height), buff).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Hyperparameters
batch_size = 256
learning_rate = 0.001
num_epochs = 10
shapes = 30
mode = 0

wandb.init(
    project='cifar-10',
    name=f'cifar_{shapes}shapes_mode{mode}',
    config={
                'epochs': num_epochs,
                'learning_rate': learning_rate,
                'model': '3-layer CNN',
                'dataset': 'cifar-10'
            }
)

# Define transforms with CIFAR-10 standard normalization.
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor in [0,1]
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
])

# Directories for the training and test images.
train_dir = '../datasets/cifar_svg/train_cifar10'
test_dir = '../datasets/cifar_svg/test_cifar10'

# Create datasets and dataloaders.
train_dataset = CIFAR10SVGDataset(root_dir=train_dir, shapes=shapes, mode=mode, transform=transform)
print('image file num: '+str(len(train_dataset)))
test_dataset = CIFAR10SVGDataset(root_dir=test_dir, shapes=shapes, mode=mode, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a three-layer convolutional network.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: input 3 channels -> 32 feature maps.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 -> 64 feature maps.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Third convolutional layer: 64 -> 128 feature maps.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Fully connected layer: from flattened features to 10 classes.
        # After three rounds of 2x2 max pooling, the 32x32 image becomes 4x4.
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        # Conv layer 1 + ReLU + 2x2 Max Pooling: 32x32 -> 16x16.
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # Conv layer 2 + ReLU + 2x2 Max Pooling: 16x16 -> 8x8.
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Conv layer 3 + ReLU + 2x2 Max Pooling: 8x8 -> 4x4.
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        # Flatten the feature maps.
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Setup device, model, loss function, and optimizer.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    wandb.log({'epoch': epoch+1, 'loss': avg_loss})
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation on the test set.
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

wandb.log({'test_acc': 100 * correct / total})
print(f"Test Accuracy: {100 * correct / total:.2f}%")
