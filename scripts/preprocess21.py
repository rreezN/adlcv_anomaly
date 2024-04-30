import os
import numpy as np
import nibabel
import random
import shutil
from tqdm import tqdm
# Dataset storage location
data_path = './data/brats21/BraTS2021_Training_Data'

# Location to save preprocessed data
save_path = './data/brats21/processed'

# if not os.path.exists(save_path):
#     os.mkdir(save_path)

dir_list = os.listdir(data_path)

modalities = ['t1', 't1ce', 't2', 'flair', 'seg']

def preprocess():
    num = 1

    for dir in (pbar:=tqdm(dir_list, total=len(dir_list))):
        
        # Skip if the directory contains this file (Brats2021 dataset specific)
        if dir == '.DS_Store':
            continue

        # print(f"{index} / {len(dir_list)}")

        patient_path = os.path.join(data_path, dir)

        model_data = {}
        for model in modalities:
            filename = dir + "_" + model + ".nii.gz"
            file_path = os.path.join(patient_path, filename)
            data = nibabel.load(file_path).get_fdata()
            model_data[model] = data

        for i in range(80, 129):
            pbar.set_description(f"Processing {dir} slice {i}")
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
            pbar.set_postfix({"num": num})


def split_data():
    # Set paths and directory names
    training_path = os.path.join(save_path, "training")
    testing_path = os.path.join(save_path, "testing")

    # Create directories for training and testing sets
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(testing_path, exist_ok=True)

    # Get list of files
    dir_list = os.listdir(save_path)[:-2]
    
    # Calculate the number of samples for training and testing sets
    total_samples = len(dir_list)
    nr_patients = int(total_samples / 49)
    patient_idx = list(range(0, nr_patients))

    # Exclude the 'training' and 'testing' directories created earlier
    train_samples = int(0.9 * nr_patients)
    test_samples = nr_patients - train_samples

    # Shuffle the list of files
    random.shuffle(patient_idx)

    train_health_num = 0
    test_health_num = 0

    for i in (pbar:= tqdm(range(nr_patients), total=nr_patients)):
        file_idx_range = list(range(patient_idx[i] * 49, (patient_idx[i] + 1) * 49))
        
        for j in file_idx_range:
            pbar.set_description(f"Patient {i}, slice {j}")
            dir_name = dir_list[j]
            source_dir_path = os.path.join(save_path, dir_name)

            file = os.listdir(source_dir_path)
            # Extract label to determine health status
            seg_files = [file_name for file_name in file if "seg" in file_name]
            if len(seg_files) == 0:
                print("---")
            seg_file = os.path.join(source_dir_path, seg_files[0])
            image = nibabel.load(seg_file).get_fdata()
            if i < train_samples:
                if image.max() == 0:
                    train_health_num += 1
                destination_dir_path = os.path.join(training_path, dir_name)
            else:
                if image.max() == 0:
                    test_health_num += 1
                destination_dir_path = os.path.join(testing_path, dir_name)
            # Move directories
            shutil.move(source_dir_path, destination_dir_path)

    print(f"Training set: Healthy {train_health_num}, Abnormal {train_samples*49 - train_health_num}, Total {train_samples*49}")
    print(f"Testing set: Healthy {test_health_num}, Abnormal {test_samples*49 - test_health_num}, Total {test_samples*49}")

if __name__ == '__main__':
    random.seed(42)

    # preprocess()

    split_data()