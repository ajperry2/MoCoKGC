from torch.utils.data import Dataset
from pathlib import Path

class MoCoKGDataset(Dataset):
    name: str
    path: str

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
    
    def clean(self):
        raise NotImplementedError
    
    def download(self):
        raise NotImplementedError
    
    def find_neighbors(self, entity):
        raise NotImplementedError
