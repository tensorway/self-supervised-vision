#%%
from numpy import convolve
import torch as th
from torchvision import transforms
from torchvision import transforms as T
import torch.nn.functional as F


class MaskedAutoencoder:
    def __init__(self, masking_proportion=0.75, patch_size=4) -> None:
        self.masking_proportion = masking_proportion
        self.patch_size = patch_size


    def process_batch(self, model, data_dict, device):
        # two randomly augmented versions of x
        y_a, y_b = data_dict['a'].to(device), data_dict['b'].to(device)

        # compute embeddings
        z_a = model.embed(y_a) # NxD
        z_b = model.embed(y_b) # NxD

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        # cross-correlation matrix
        N, D = z_a_norm.shape
        c = ( z_a_norm.T @ z_b_norm )/ N # DxD
        c = th.nan_to_num(c, 0)

        # loss
        c_diff = (c - th.eye(D, device=device))**2 # DxD

        #calculate lambda
        if self.auto_lambda:
            self.lambda_ = 1/N

        # multiply off-diagonal elems of c_diff by lambda
        off_diagonal_mul = (th.ones_like(c_diff, device='cpu') - th.eye(D))*self.lambda_ + th.eye(D)
        off_diagonal_mul = off_diagonal_mul.to(device)
        loss = (c_diff * off_diagonal_mul).sum()

        # create the debug dict
        debug_dict = {name:value for name, value in zip(vars().keys(), vars().values())}

        return loss, debug_dict

    def get_hard_transform(self):
        return T.Compose([
            T.Resize(48),
            T.RandomCrop(32),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            T.RandomGrayscale(p=0.3),
            T.RandomSolarize(threshold=200, p=0.3),
            T.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=3), # seems to hurt perfomance
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])


def patchify(imgs, patch_size: int):
    b, c, w, h = imgs.shape
    imgs = imgs.view(b, c, -1).transpose(2, 0, 1)

def mask(imgs):
    # b, c, w, h
    patches = imgs.view(b, c, patch_size, -1)
    index = th.range()

def _get_conv_weight(h, w):
    zeros = []
    for i in range(3*h*w):
        z = th.zeros((3*h*w,))
        z[i] = 1
        zeros.append(z.reshape(1, 3, h, w))
    return th.cat(zeros)

def patchify(imgs, patch_size):
    '''
    returns b, -1, patch_size x patch_size
    '''
    weight = _get_conv_weight(patch_size, patch_size)
    convolved = F.conv2d(imgs, weight, None, patch_size)
    return convolved.view(imgs.shape[0], 3*patch_size*patch_size, -1).permute(0, 2, 1)

# %% 
if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    import matplotlib.pyplot as plt
    dataset = CIFAR10('../data', True, transform=T.ToTensor())
    patch_size = 16
    imgs = dataset[0][0][None]
    patches = patchify(imgs, patch_size)
    patches = patches.view(-1, 3, patch_size, patch_size).permute(0, 2, 3, 1)
    print(patches.shape)
    


    pilimg = dataset[0][0].cpu().permute(1, 2, 0).numpy()
    plt.imshow(pilimg)
# %%
for i, patch in enumerate(patches):
    plt.subplot(2, 2, i+1)
    plt.imshow(patch.numpy())
# %%
