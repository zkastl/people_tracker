import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import CNN

# Define the CNN architecture
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 4 * 4, 512)
#         self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 4 * 4)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model and move it to the device
model = CNN().to(device)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Training loop
# def train(epochs):
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data[0].to(device), data[1].to(device)

#             model.optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = model.criterion(outputs, labels)
#             loss.backward()
#             model.optimizer.step()

#             running_loss += loss.item()
#             if i % 100 == 99:    # print every 100 mini-batches
#                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
#                 running_loss = 0.0

#     print('Finished Training')

# Evaluation function
# def evaluate():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print(f'Accuracy on test images: {100 * correct / total}%')

# Train the model
model.train(device=device, trainloader=trainloader, epochs=10)

# Evaluate the model
model.evaluate(device=device, testloader=testloader)

# save the model
model.save_model()