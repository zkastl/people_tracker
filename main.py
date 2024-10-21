import time
import torch
import torchvision
import torchvision.transforms as transforms
import tracemalloc
import torch.nn as nn

from models import CNN, MobileNetV2

BYTE_TO_MEGABYTE = 1048576

def main(model:nn.Module, train:bool=True, batch_size=64):

    # Set device
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    # Create the model and move it to the device
    nn_model = model.to(device)

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Train the model
    if train:
        t_start = time.time()
        nn_model.train(device=device, trainloader=trainloader, epochs=10)
        t_end = time.time()
        tr_time = t_end - t_start
        print(f'Total training time: {tr_time:.2f} seconds')
    
    else:
        tr_time = float('nan')

    
    # Evaluate the model
    t_start = time.time()
    accuracy = model.evaluate(device=device, testloader=testloader)
    t_end = time.time()
    if_time = t_end - t_start
    print(f'Total evaluation time: {if_time:.2f} seconds')

    # save the model
    model.save_model()

    return device, tr_time, if_time, batch_size, accuracy

def track_model(model, train:bool=True, batch_size=64, device=None):

    tracemalloc.start()
    device, tr_time, if_time, bs, accuracy = main(model, train, batch_size=batch_size)

    # show how much RAM the above code allocated and the peak usage
    current, peak =  tracemalloc.get_traced_memory()
    print(f"Current: {current:0.2f}, Peak: {peak:0.2f}")
    tracemalloc.stop()

    return f'| {model.identifier} | {device} | {tr_time:.2f} | {if_time:.2f} | {batch_size} | {(peak // BYTE_TO_MEGABYTE):0.2f} | {accuracy} | CIFAR-10 |'

# ENTRY POINT
if __name__ == "__main__":
    
    models = [
        (CNN(), True, 8),
        (MobileNetV2(), True, 64)
        ]
    
    results = []

    for model in models:
        results.append(track_model(model[0], model[1], model[2]))

    for result in results:
        print(result)