from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DATA(Dataset):
    def __init__(self, data_x, data_y):
        self.data = data_x
        self.out = data_y

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' get data '''
        serie = self.data[idx]
        cls = self.out[idx]

        return self.transform(serie), self.transform(cls)