# TODO(ojasw): write code to test vgg19 checkpoints against numerosity input
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import scipy.optimize
import argparse
from tqdm import tqdm


# --- Helper functions from the provided code ---
def load_model(model_name, checkpoint_path=None):
    if model_name == "vgg19":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "vgg19", weights=None
        )  # Load without pretrained weights
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if "model_state_dict" in checkpoint:
                # Load from your custom training format
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Load from a standard pretrained model or a different checkpoint format
                model.load_state_dict(checkpoint)
        return model
    else:
        print("Invalid model name. Available models are [vgg19].")
        return None

# for natural images
def load_data_from_folder(folder_path, numerosity):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    images = []
    labels = []
    image_paths = glob.glob(
        os.path.join(folder_path, str(numerosity), "*")
    )  # Adjust path joining
    for f in image_paths:
        labels.append(os.path.basename(f))
        img = Image.open(f).convert("RGB")
        images.append(np.asarray(transform(img)))

    images = np.array(images)
    images = torch.Tensor(images)
    return images, labels


# for 1/1 1/2 1/3... folders
# def load_data_from_folder(folder_path, numerosity):
#     transform = transforms.Compose(
#         [
#             transforms.Resize(256),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )

#     images = []
#     labels = []
#     # Corrected glob pattern to handle nested structure
#     image_paths = glob.glob(
#         os.path.join(folder_path, str(numerosity), "*", "*.png")
#     )  # Include *.png to only load image files

#     for f in image_paths:
#         labels.append(os.path.basename(f))
#         try:
#             img = Image.open(f).convert(
#                 "RGB"
#             )  # Try opening as RGB. Might be needed for grayscale images
#             images.append(np.asarray(transform(img)))
#         except Exception as e:  # Handle potential errors gracefully
#             print(f"Error loading image {f}: {e}")

#     if not images:
#         raise ValueError(
#             f"No images found in directory: {os.path.join(folder_path, str(numerosity))}"
#         )
#     images = np.array(images)
#     images = torch.Tensor(images)

#     return images, labels


def get_activation_classifier(model, images, device):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    h = model.classifier[6].register_forward_hook(get_activation("linearlayer"))
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        images = images.to(device)
        model(images)
    h.remove()
    return activation["linearlayer"].cpu()


# --- Numerosity experiment function ---
def numerosity_experiment(model, stimuli_folder, n1, n2, M, device):
    all_similarities = []
    for _ in range(M):
        # Randomly select one image for each numerosity
        images_n1, _ = load_data_from_folder(stimuli_folder, n1)
        images_n2, _ = load_data_from_folder(stimuli_folder, n2)

        random_index_n1 = np.random.randint(0, images_n1.shape[0])
        random_index_n2 = np.random.randint(0, images_n2.shape[0])

        image_n1 = images_n1[random_index_n1].unsqueeze(0)
        image_n2 = images_n2[random_index_n2].unsqueeze(0)

        # Get activations
        act_n1 = get_activation_classifier(model, image_n1, device)
        act_n2 = get_activation_classifier(model, image_n2, device)

        # Compute cosine similarity
        similarity = cosine_similarity(act_n1.numpy(), act_n2.numpy())[0][0]
        all_similarities.append(similarity)

    return np.mean(all_similarities)


# --- Main script ---
def main(stimuli_type, checkpoints_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device: ", device)
    stimuli_folder = os.path.join("numerosity-stimuli", stimuli_type)
    M = 20 # change to 20 for final run
    all_results = {}

    print("Running numerosity experiment...")
    for epoch in tqdm(range(3, 40, 3)):
        checkpoint_path = os.path.join(checkpoints_dir, f"vgg19_epoch_{epoch}.pth")
        model = load_model("vgg19", checkpoint_path).to(device)
        epoch_results = []

        for n1 in range(1, 6):
            for n2 in range(1, 6):
                if n1 != n2:
                    avg_similarity = numerosity_experiment(
                        model, stimuli_folder, n1, n2, M, device
                    )
                    epoch_results.append((n1, n2, avg_similarity))
                    print(".", end="") # Print progress
        print(" finished epoch ", epoch)

        all_results[epoch] = epoch_results
    # Save results to a file
    print("Saving results to a file...")
    torch.save(all_results, f"results_{stimuli_type}.pth")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numerosity experiment script.")
    parser.add_argument(
        "--stimuli_type",
        type=str,
        default="equal-circles",
        help="Type of stimuli to use (e.g., equal-circles, equal-squares).",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory containing the model checkpoints.",
    )
    args = parser.parse_args()
    results = main(args.stimuli_type, args.checkpoints_dir)
    print(results)
