import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.models as models
import json
from workspace_utils import active_session


import argparse

parser = argparse.ArgumentParser('change default parameters and settings')
parser.add_argument('--save_dir', type=str, help='save_directory to /opt (default: current)')
parser.add_argument('--arch', type=str, help='use resnet101 architecture (default: resnet50)')
parser.add_argument('--learning_rate', type=float, help='change learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, help='change number of epochs (default: 4)')
parser.add_argument('--gpu', type=str, help='use gpu (y/n) if available (default: n)')

args = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(train_data)
valid_data_size = len(test_data)
test_data_size = len(valid_data)


# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Print the train, validation and test set data sizes
print ('Train size:  ' + str(train_data_size))
print ('Validation size:  ' + str(valid_data_size))
print ('Test size:  ' + str(test_data_size))

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Build the network
if args.gpu is None:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.arch is None:
    model = models.resnet50(pretrained=True)
    model_name = 'resnet50'
else:
    model = models.resnet101(pretrained=True)
    model_name = 'resnet101'

for param in model.parameters():
    param.requires_grad = False

fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(.3),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))
model.fc = fc

criterion = nn.NLLLoss()
# for resnet50
if args.learning_rate is None:
    lr_value = 0.001
else:
    lr_value = args.learning_rate
optimizer = optim.Adam(model.fc.parameters(), lr=lr_value)

model.to(device)


# Train the network
with active_session():
    if args.epochs is None:
        epochs = 4
    else:
        epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
# Save the checkpoint
if args.save_dir is None:
    torch.save({'fc':fc, 'model':model.state_dict(), 'lr': lr_value, 'epoch':epochs}, model_name)
else:
    path = os.path.join(args.save_dir, '{}_epochs{}.pth'.format(model_name, epochs))
    torch.save({'fc':fc, 'model':model.state_dict(), 'lr': lr_value, 'epoch':epochs}, path)

print ('Model saved.  Ready to predict.')
