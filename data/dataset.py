
#%%
import torch as th
from torchvision.datasets import CIFAR10
from data.transforms import val_transform


class DoubleCifar10(CIFAR10):
    def __init__(self, root: str = './data', train: bool = True, transform = None, target_transform = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
    
    def __getitem__(self, index: int):
        a = super().__getitem__(index)[0]
        b = super().__getitem__(index)[0]
        return {'a':a, 'b':b}
#%%
if __name__ == '__main__':
    trainset = DoubleCifar10(transform=val_transform)
    train_dataloader = th.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=6)
# %%
