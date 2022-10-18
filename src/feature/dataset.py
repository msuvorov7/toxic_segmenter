from torch.utils.data import Dataset


class ToxicDataset(Dataset):
    def __init__(self, features: list, tags: list):
        self.features = features
        self.tags = tags

    def __getitem__(self, item) -> dict:
        return {
            'feature': self.features[item],
            'tag': self.tags[item]
        }

    def __len__(self) -> int:
        return len(self.tags)
