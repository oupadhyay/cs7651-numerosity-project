import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from torch.amp import GradScaler, autocast
# from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn

print("Starting VGG19 training...")
with open("vgg.out", "a") as f:
    f.write("Starting VGG19 training...\n")

# 1. Model Definition (VGG19 from scratch - no pretrained weights)
model = torchvision.models.vgg19(weights=None)

# 2. Hyperparameters
learning_rate = 1e-2
num_epochs = 40
batch_size = 256  # Increased batch size for H100
num_workers = 1  # Increased number of workers for faster data loading
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 3. Data Loading and Preprocessing (ImageNet)
# Using faster data loading with persistent_workers and pin_memory
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Replace 'path/to/imagenet' with the actual path to your ImageNet dataset
train_dataset = torchvision.datasets.ImageNet(
    "./imagenet", split="train", transform=train_transforms
)
val_dataset = torchvision.datasets.ImageNet(
    "./imagenet", split="val", transform=val_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Pin memory for faster transfer to GPU
    persistent_workers=True,  # Use persistent workers for faster data loading between epochs
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

# 4. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4
)

# 5. GPU Setup (with AMP and Data Parallelism if available)
device = torch.device("cuda")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    with open("vgg.out", "a") as f:
        f.write(f"Using {torch.cuda.device_count()} GPUs!\n")
    model = nn.DataParallel(model)
model.to(device)
cudnn.benchmark = True  # Enable cuDNN autotuner to find best algorithms
scaler = GradScaler('cuda')  # For mixed precision training

# 6. Checkpoint Loading and Resuming
start_epoch = 0
if os.listdir(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1][:-4]))
        checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
        try:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"Resuming training from epoch {start_epoch}")
            with open("vgg.out", "a") as f:
                f.write(f"Resuming training from epoch {start_epoch}\n")
        except:
            print("Couldnt load from checkpoint. Starting from scratch")
            with open("vgg.out", "a") as f:
                f.write("Couldnt load from checkpoint. Starting from scratch\n")
    else:
        print("Couldnt load from checkpoint. Starting from scratch")
        with open("vgg.out", "a") as f:
            f.write("Couldnt load from checkpoint. Starting from scratch\n")
else:
    print("Couldnt load from checkpoint. Starting from scratch")
    with open("vgg.out", "a") as f:
        f.write("Couldnt load from checkpoint. Starting from scratch\n")

# 7. Training Loop with Mixed Precision
print("Starting training loop...")
with open("vgg.out", "a") as f:
    f.write("Starting training loop...\n")
for epoch in range(start_epoch, num_epochs):
    with open("vgg.out", "a") as f:
        f.write(f"Starting epoch {epoch+1}\n")
    model.train()
    running_loss = 0.0

    print(f"Starting epoch {epoch+1}")
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast('cuda'):  # Enable mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()  # Scale loss and backward
        scaler.step(optimizer)  # Optimizer step
        scaler.update()  # Update scaler

        running_loss += loss.item() * images.size(0)
        with open("vgg.out", "a") as f:
            f.write(".")
    with open("vgg.out", "a") as f:
        f.write("\n")

    epoch_loss = running_loss / len(train_dataset)
    print(f"\nEpoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    with open("vgg.out", "a") as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}\n")

    # 8. Save Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"vgg19_epoch_{epoch+1}.pth")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")
    with open("vgg.out", "a") as f:
        f.write(f"Checkpoint saved to {checkpoint_path}\n")

print("Training finished!")
with open("vgg.out", "a") as f:
    f.write("Training finished!\n")
