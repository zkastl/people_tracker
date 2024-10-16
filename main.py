import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import PIL
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes, adjust if needed

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model and move it to the device
model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def unpickle(file):
    import pickle
    with open(file, 'rb') as data:
        dict = pickle.load(data, encoding='bytes')
    return dict

# Placeholder for dataset loading
# Replace with your actual dataset
# trainset = YourDataset(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# Training loop (placeholder)
def train(epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

# Uncomment to train
# train(epochs=5)

# Evaluation (placeholder)
def evaluate():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total}%')

# Uncomment to evaluate
# evaluate()

def main():
    file_name = '/home/zak/Code/kastl-phd-Coursework/test/cifar-10-batches-py/data_batch_1'
    data = unpickle(file_name)
    img_0 = data[b'data'][0]
    reshaped = np.array(img_0.reshape(3,32,32))
    reshaped_t = reshaped.transpose(1,2,0)

    # Normalize pixel values if they are in the range 0-255
    if reshaped_t.max() > 1:
        reshaped_t = reshaped_t / 255.0

    plt.imshow(reshaped_t)
    plt.axis('off')
    plt.show()

    a = 1

main()