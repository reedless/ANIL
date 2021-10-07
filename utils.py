from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)