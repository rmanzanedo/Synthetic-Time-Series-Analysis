rom
torch.utils.data
import Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


class DATA(Dataset):
    def __init__(self, data_x):
        self.data = data_x.to_numpy(dtype=np.double)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' get data '''
        serie = self.data[idx]

        return serie
