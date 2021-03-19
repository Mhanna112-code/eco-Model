from scipy import misc
import torch
from torch.utils.data import Dataset, DataLoader


class SomeImageDataset(Dataset):
    """The training table dataset.
    """

    def __init__(self, x_path):
        x_filenames = glob(x_path + '*.png')  # Get the filenames of all training images

        self.x_data = [torch.from_numpy(misc.imread(filename)) for filename in
                       x_filenames]  # Load the images into torch tensors
        self.y_data = target_label_list  # Class labels
        self.len = len(self.x_data)  # Size of data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len