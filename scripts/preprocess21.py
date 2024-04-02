import os
import numpy as np
import nibabel
import random
import shutil

# Dataset storage location
data_path = './data/brats21/BraTS2021_Training_Data'

# Location to save preprocessed data
save_path = './data/brats21/processed/training'

# if not os.path.exists(save_path):
#     os.mkdir(save_path)

dir_list = os.listdir(data_path)

modalities = ['t1', 't1ce', 't2', 'flair', 'seg']

def preprocess():
    num = 1

    for index, dir in enumerate(dir_list):
        
        # Skip if the directory contains this file (Brats2021 dataset specific)
        if dir == '.DS_Store':
            continue

        print(f"{index} / {len(dir_list)}")

        patient_path = os.path.join(data_path, dir)

        model_data = {}
        for model in modalities:
            filename = dir + "_" + model + ".nii.gz"
            file_path = os.path.join(patient_path, filename)
            data = nibabel.load(file_path).get_fdata()
            model_data[model] = data

        for i in range(80, 129):
            file_num = str(num).zfill(6)
            save_slice_path = os.path.join(save_path, file_num)
            if not os.path.exists(save_slice_path):
                os.mkdir(save_slice_path)

            for model in modalities:
                file_name = dir + "_" + model + "_" + str(i).zfill(3) + ".nii.gz"

                save_model_path = os.path.join(save_slice_path, file_name)

                if model == 'seg':
                    # Map label values: 0, 1, 2, 4 to 0, 1, 2, 3
                    label = model_data[model][..., i]
                    label[label == 4] = 3
                    label = nibabel.Nifti1Image(label, affine=np.eye(4))
                    nibabel.save(label, save_model_path)
                else:
                    img_data = model_data[model]
                    x = img_data[..., i] - np.nanmin(img_data[..., i])
                    y = np.nanmax(img_data[..., i]) - np.nanmin(img_data[..., i])
                    y = y if y != 0 else 1.0
                    img = x / y  # (240, 240)

                    if img.max() > 1.0 or img.min() < 0:
                        print(f"--Error: {num} --")

                    img = nibabel.Nifti1Image(img, affine=np.eye(4))
                    nibabel.save(img, save_model_path)
            num += 1


def split_data():
    # Set paths and directory names
    training_path = os.path.join(save_path, "training")
    testing_path = os.path.join(save_path, "testing")

    # Create directories for training and testing sets
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(testing_path, exist_ok=True)

    # Get list of files
    dir_list = os.listdir(save_path)

    # Shuffle the list of files
    random.shuffle(dir_list)

    # Calculate the number of samples for training and testing sets
    total_samples = len(dir_list)
    
    # Exclude the 'training' and 'testing' directories created earlier
    total_samples = total_samples - 2
    train_samples = int(0.9 * total_samples)
    test_samples = total_samples - train_samples

    train_health_num = 0
    test_health_num = 0

    for i, dir_name in enumerate(dir_list):
        print(f"{i} / {total_samples}")

        if dir_name == 'training' or dir_name == 'testing':
            continue

        source_dir_path = os.path.join(save_path, dir_name)

        if i < train_samples:
            file = os.listdir(source_dir_path)

            # Extract label to determine health status
            seg_files = [file_name for file_name in file if "seg" in file_name]

            if len(seg_files) == 0:
                print("---")
            seg_file = os.path.join(source_dir_path, seg_files[0])
            image = nibabel.load(seg_file).get_fdata()

            if image.max() == 0:
                train_health_num += 1

            destination_dir_path = os.path.join(training_path, dir_name)

        else:
            file = os.listdir(source_dir_path)

            seg_files = [file_name for file_name in file if "seg" in file_name]
            if len(seg_files) == 0:
                print("---")
            seg_file = os.path.join(source_dir_path, seg_files[0])
            image = nibabel.load(seg_file).get_fdata()
            if image.max() == 0:
                test_health_num += 1
            destination_dir_path = os.path.join(testing_path, dir_name)

            # Extract 'seg' for test_labels separately
            # ... (write your code here)

        # Move directories
        shutil.move(source_dir_path, destination_dir_path)

    print(f"Training set: Healthy {train_health_num}, Abnormal {train_samples - train_health_num}, Total {train_samples}")
    print(f"Testing set: Healthy {test_health_num}, Abnormal {test_samples - test_health_num}, Total {test_samples}")

if __name__ == '__main__':
    preprocess()

    split_data()