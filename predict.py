import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision.models as models
import json

import argparse
parser = argparse.ArgumentParser('define filename, path and other settings')
parser.add_argument('filename', help='filename of trained model')
parser.add_argument('--file_dir', type=str, help='directory of trained model either /opt or (default: current)')
parser.add_argument('--arch', type=str, help='specify if resnet101 architecture (default: resnet50)')
parser.add_argument('--K', type=int, help='to K most likely classes (default: 5)')
parser.add_argument('--gpu', type=str, help='use gpu (y/n) if available (default: n)')

args = parser.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

if args.gpu is None:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
def loadModel(checkpoint):
    saved_model = torch.load(checkpoint)
    if args.arch is None:
        model = models.resnet50(pretrained=True)
        model_name = 'resnet50'
    else:
        model = models.resnet101(pretrained=True)
        model_name = 'resnet101'
    model.fc = saved_model['fc']
    model.load_state_dict(saved_model['model'])
    optimizer = optim.Adam(model.parameters(), lr = saved_model['lr'])
    
    return model, optimizer


if args.file_dir is None:
    model, optimizer = loadModel(args.filename)
else:
    path = args.file_dir
    filename = os.path.join(path, args.filename)
    
if args.arch is None:
    model = models.resnet50(pretrained=True)
    model_name = 'resnet50'
else:
    model = models.resnet101(pretrained=True)
    model_name = 'resnet101'

# Do the image processing
def process_image(image):
    image = Image.open(image_path)
    image = image.resize((256,256))
    image = image.crop((16, 16, 240, 240))
    np_image = np.array(image)/255
    np_image = (np_image -(np.array([0.485, 0.456, 0.406])))/(np.array([0.229,0.224,0.225]))
    np_image = np_image.transpose(2,0,1)
    
    return np_image


# Convert a PyTorch tensor and display it
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Perform class predition
if args.K is None:
    topk_value = 5
else:
    topk_value = args.K

def predict(image_path, model, topk=topk_value):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # predict the class from an image file
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        prediction = model.forward(image.view(1,3,224,224).to(device).float())
        prob, classes = torch.exp(prediction).topk(topk_value,dim = 1)
        classes = classes[0,:] 
        label = []
        for c in classes:
            c = (cat_to_name[str(c.item())])
            label.append(c)
    
    return prob, label

# Display an image along with the top 5 classes
image_path = ('flowers/test/1/image_06760.jpg')
prob, label = predict(image_path, model)

sorted_label = [label for _,label in sorted(zip(prob[0,:],label))]
sorted_prob, indices = torch.sort(prob.to(device), dim = 1)

fig,ax = plt.subplots(2,1, figsize = (10,10))
with  Image.open(image_path) as image:
    ax[0].imshow(image)
ax[0].set_title(sorted_label[4])
ax[1].barh(np.arange(len(sorted_label)), sorted_prob[0,:], align = 'center')
ax[1].set_xlabel(xlabel = 'Probability')
ax[1].set_ylabel(ylabel = 'category')
ax[1].set_yticks(np.arange(len(sorted_label)))
ax[1].set_yticklabels(sorted_label)
