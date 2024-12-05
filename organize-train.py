import tarfile
import os
import shutil
import glob

train_tar_dir = "./train"
output_train_dir = "./imagenet/train"


def organize_train_data(tar_dir, output_dir):
    """Organizes ImageNet training data, handling interruptions."""

    os.makedirs(output_dir, exist_ok=True)

    tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))

    for tar_file in tar_files:
        wnid = os.path.splitext(os.path.basename(tar_file))[0]
        class_folder = os.path.join(output_dir, wnid)
        os.makedirs(class_folder, exist_ok=True)
        with open("log.txt", "a") as log_file:
            log_file.write(f"Processing: {tar_file}\n")

        extracted_files = set(os.listdir(class_folder))  # Files already extracted

        try:
            with tarfile.open(tar_file, "r") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".JPEG"):
                        filename = os.path.basename(member.name)
                        if (
                            filename not in extracted_files
                        ):  # Check if already extracted
                            try:
                                tar.extract(member, path=class_folder)
                                extracted_files.add(
                                    filename
                                )  # Add to the set of extracted files
                                with open("log.txt", "a") as log_file:
                                    log_file.write(f"Extracted: {filename} to {class_folder}\n")
                            except Exception as e:
                                print(
                                    f"Error extracting {filename} from {tar_file}: {e}"
                                )
        except (
            Exception
        ) as e:  # Handles corrupt tar files or interruptions during opening
            print(f"Error processing {tar_file}: {e}")


with open("log.txt", "a") as log_file:
    log_file.write("Organizing training images...\n")
organize_train_data(train_tar_dir, output_train_dir)
with open("log.txt", "a") as log_file:
    log_file.write("Training images organized (or resumed).\n")
