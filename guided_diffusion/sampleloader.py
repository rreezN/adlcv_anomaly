import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SampledDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if "output" in filename and filename.endswith(".npy")]

    def __len__(self):
        return len(self.image_paths) // 4

    def __getitem__(self, idx):
        image_stack = []
        for i in range(4):
            image_path = self.image_paths[idx * 4 + i]
            image = np.load(image_path)
            image_stack.append(torch.tensor(image).float())
        stck = torch.stack(image_stack)
        return (torch.stack(image_stack))