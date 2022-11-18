
import typing
import torch


class DictionaryDataset(torch.utils.data.Dataset):
    """Wrapper for data in dictionaries."""
    def __init__(self, data: dict, exclude_keys: typing.List[str] = []):
        super(DictionaryDataset, self).__init__()
        
        self.data = data
        self.exclude_keys = exclude_keys
        self.keys = [c for c in self.data if c not in exclude_keys]

    def __getitem__(self, index: int) -> dict:
        out = dict()
        for k in self.data:
            if k in self.exclude_keys:
                continue
            out[k] = self.data[k][index]
        
        return out

    def __len__(self):
        return self.data[self.keys[0]].__len__()
