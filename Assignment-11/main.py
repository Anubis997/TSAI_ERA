# -*- coding: utf-8 -*-
"""resnet_18/34_models.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JkLulp6hUqTIbO3RrzT2ltsiwMM9Qgmn
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.datasets as datasets
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt

!git clone https://github.com/Anubis997/TSAI_ERA

cd /content/TSAI_ERA/

cd Assignment-11/

"""# Loading model"""

from Resnet_18 import Layer
model=Layer()

"""# Loading transformations"""

cd /content/TSAI_ERA/

from utils import get_train_test_transform

train_transforms, test_transforms = get_train_test_transform()

"""# Test and Train transforms"""

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)


# Define data loaders
SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)
if cuda:
    torch.cuda.manual_seed(SEED)

# Dataloader arguments
dataloader_args = dict(shuffle=True, batch_size=64, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# Train dataloader
train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

# Test dataloader
test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

# Move model to GPU if available
device = torch.device("cuda" if cuda else "cpu")
# Example: Assuming 'model' is your defined neural network model
# model = YourModel()
# model.to(device)

for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)

"""# Range Test"""

!pip install torch_lr_finder
from torch_lr_finder import LRFinder
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=0.1, num_iter=100, step_mode="linear")
lr_finder.plot(log_lr=False)
lr_finder.reset()

"""# Calibrating Hyperparameters"""

# Set LRMAX and LRMIN (example values)
LRMAX =  4.40E-02  # Example value
LRMIN = LRMAX / 10  # Example value, LRMIN as 1/10th of LRMAX

# Calculate the total number of steps (batches) per epoch

total_epochs = 20
total_steps = total_epochs * len(train_loader)
optimizer = optim.Adam(model.parameters(), lr=LRMIN)
scheduler = OneCycleLR(optimizer, max_lr=LRMAX, total_steps=total_steps, pct_start=5/24,div_factor=10,final_div_factor=10)

"""# Importing test and train"""

cd /content/TSAI_ERA/

from train_and_test import train, test

"""# Running the model"""

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

lr_list=[]
for epoch in range(20):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer,scheduler, epoch)
    lr_list.append(scheduler.get_last_lr())
    print(scheduler.get_last_lr())
    test(model, device, test_loader)

"""# Misclassified images"""

cifar_10_labels = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

reverse_cifar_10_labels = {v: k for k, v in cifar_10_labels.items()}

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

transform_denormalize = transforms.Compose([
    transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
    transforms.ToPILImage()
])


def get_misclassified_images(model, testset, number):
    model.eval()

    # List to store misclassified images and their details
    misclassified_data = []

    with torch.no_grad():
        for i in range(len(testset)):
            image, label = testset[i]
            image = image.unsqueeze(0).cuda()  # Add batch dimension and move to GPU if available
            output = model(image)
            _, predicted = torch.max(output, 1)

            if predicted != label:
                # Convert tensor image to PIL Image for visualization
                image_pil = transform_denormalize(image.squeeze().cpu())

                # Append misclassified image, original label, predicted label, and image itself
                misclassified_data.append({
                    'original_label': testset.classes[label],
                    'predicted_label': testset.classes[predicted],
                    'image': image_pil
                })

            if len(misclassified_data) >= number:
                break

    return misclassified_data[:number]

misclassified_data = get_misclassified_images(model, testset, 10)

pip install grad-cam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
import cv2
# Assuming 'model' is your ResNet-18 model and 'data' is the misclassified image data structure
model.cuda()
model.eval()

# Define target layer
target_layer = [model.layer4[-1]]

for i, data in enumerate(misclassified_data):

    # Assuming data['image'] is a PIL Image
    img = data['image']
    original_label = data['original_label']
    predicted_label = data['predicted_label']

    # Define transformations to be applied to the image
    img = cv2.resize(np.array(img), (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add batch dimension and move to GPU

    # Instantiate GradCAM
    cam = GradCAM(model=model, target_layers=target_layer)

    #target images
    targets = [ClassifierOutputTarget(reverse_cifar_10_labels.get(original_label, -1))]

    # Generate Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor)


    # # Visualize the heatmap on the image
    visualization = show_cam_on_image(img, grayscale_cam[0,:], use_rgb=True)

    # # Display the visualization
    plt.figure(figsize=(4, 4))
    print("Original Label:", original_label)
    print("Predicted Label:", predicted_label)
    plt.imshow(visualization)
    plt.axis('off')
    plt.show()

