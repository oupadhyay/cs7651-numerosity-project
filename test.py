import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
import numpy as np

# Load the VGG19 model and the pretrained model
vgg19 = models.vgg19()
pretrained_vgg19 = models.vgg19(pretrained=True)

# Load the checkpoint
checkpoint = torch.load("./checkpoints/vgg19_epoch_25.pth")
vgg19.load_state_dict(checkpoint["model_state_dict"])

# Prepare the ImageNet-1K dataset
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

imagenet_data = ImageNet(root="./imagenet", split="val", transform=transform)
data_loader = DataLoader(imagenet_data, batch_size=1, shuffle=True)


# Function to calculate accuracy
def calculate_accuracy(model, data_loader, shots):
    model.eval()
    correct_1_shot = 0
    correct_5_shot = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i >= shots:
                break
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_1_shot += (predicted == labels).sum().item()

            _, top5_pred = outputs.topk(5, 1, True, True)
            correct_5_shot += (labels.view(-1, 1) == top5_pred).sum().item()
            total += labels.size(0)

    accuracy_1_shot = correct_1_shot / total
    accuracy_5_shot = correct_5_shot / total
    return accuracy_1_shot, accuracy_5_shot


# Calculate 1-shot and 5-shot accuracy for the model and pretrained model
shots = 100
vgg19_1_shot_acc, vgg19_5_shot_acc = calculate_accuracy(vgg19, data_loader, shots)
pretrained_1_shot_acc, pretrained_5_shot_acc = calculate_accuracy(
    pretrained_vgg19, data_loader, shots
)

# Print the results
print(
    f"VGG19 Model - 1-shot accuracy: {vgg19_1_shot_acc:.4f}, 5-shot accuracy: {vgg19_5_shot_acc:.4f}"
)
print(
    f"Pretrained VGG19 Model - 1-shot accuracy: {pretrained_1_shot_acc:.4f}, 5-shot accuracy: {pretrained_5_shot_acc:.4f}"
)
