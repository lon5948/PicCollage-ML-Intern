import torch
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data
        self.y_data = torch.FloatTensor(y_data)
        self.transform = transform  # transformation for the image

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        image = self.x_data[index]
        corr = self.y_data[index]
        if self.transform is not None:
            image = self.transform(image)  
        return image, corr
    