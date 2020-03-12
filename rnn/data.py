from torch.utils.data import Dataset
import numpy as np



class DATA(Dataset):
    def __init__(self, data_x, data_y):
        self.data = data_x.to_numpy(dtype=np.double)
        self.out = data_y.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        serie = self.data[idx]
        cls = self.out[idx]

        return serie, cls
