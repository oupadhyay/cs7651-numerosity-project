import tarfile
import os
import shutil
from scipy.io import loadmat

val_tar_path = "./ILSVRC2012_img_val.tar"
val_ground_truth_path = "./ILSVRC2012_validation_ground_truth.txt"
meta_mat_path = "./meta.mat"
output_val_dir = "./imagenet/val"


def organize_val_data(tar_path, ground_truth_path, meta_path, output_dir):
    """Organizes ImageNet validation data into class folders."""

    # Create the output validation directory
    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth labels
    with open(ground_truth_path, "r") as f:
        val_labels = [line.strip() for line in f]

    # Load meta data (mapping from labels to wnids)
    meta_data = loadmat(meta_path, struct_as_record=False, squeeze_me=True)["synsets"]
    label_to_wnid = {
        idx: meta_data[idx - 1].WNID for idx in range(1, len(meta_data) + 1)
    }

    # Extract images from tar and move to class folders
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".JPEG"):
                filename = os.path.basename(member.name)
                try:
                    # Extract the image index from the filename
                    image_index = int(filename.split("_")[-1].split(".")[0])
                except ValueError:
                    print(f"Skipping file {filename} due to incorrect format.")
                    continue  # Skip files that don't match the expected format

                # Get the label index and map to wnid
                label_index = int(val_labels[image_index - 1])  # Labels are 1-indexed
                wnid = label_to_wnid[label_index]

                # Create class folder (wnid)
                class_folder = os.path.join(output_dir, wnid)
                os.makedirs(class_folder, exist_ok=True)

                # Extract and move the image
                tar.extract(member, path=output_dir)
                shutil.move(
                    os.path.join(output_dir, member.name),
                    os.path.join(class_folder, filename),
                )


# Run the function to organize
organize_val_data(val_tar_path, val_ground_truth_path, meta_mat_path, output_val_dir)
print("Validation images organized into class folders.")
