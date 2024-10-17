import time
import torch
import torchvision
import torchvision.transforms as transforms

from model import CNN

if __name__ == "__main__":

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'

    # Create the model and move it to the device
    model = CNN().to(device)

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

    # Train the model
    t_start = time.time()
    model.train(device=device, trainloader=trainloader, epochs=10)
    t_end = time.time()
    print(f'Total evaluation time: {t_end - t_start:.2f} seconds')


    # Evaluate the model
    t_start = time.time()
    model.evaluate(device=device, testloader=testloader)
    t_end = time.time()
    print(f'Total evaluation time: {t_end - t_start:.2f} seconds')

    # save the model
    model.save_model()