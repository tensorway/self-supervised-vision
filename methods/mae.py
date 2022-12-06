#%%
import torch as th
import torch
from torchvision import transforms as T
import torch.nn.functional as F
from data.transforms import reverse_normalize_transform
import math

class MaskedAutoencoderProcessor:
    def __init__(self, masking_proportion=0.75, patch_size=4, img_size=(32, 32)) -> None:
        self.masking_proportion = masking_proportion
        self.patch_size = patch_size
        self.n_patches_in_row = img_size[0] // patch_size

    def process_batch(self, model, data_dict, device):
        imgs = data_dict['a'].to(device)
        patches = imgs_to_patches(imgs, patch_size=self.patch_size)
        tokens = model.backbone.patches_to_tokens(patches)

        tokens, kept_idxs, not_kept_idxs = drop_tokens(tokens, self.masking_proportion)
        predictions = model.backbone.reconstruct(tokens)
        mask = get_masked_img_in_patch_form(torch.ones_like(patches), not_kept_idxs)

        # mean loss on masked patches
        loss = ((predictions-patches)**2 * mask).mean() / self.masking_proportion

        debug_dict = {name:value for name, value in zip(vars().keys(), vars().values())}

        return loss, debug_dict

    def get_hard_transform(self):
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def tensorboard_plot(self, writer, step, debug_dic, n_rows=10, n_columns=10):
        imgs = []
        for i in range(n_columns):
            pred_patches = debug_dic['predictions'][n_rows*i:n_rows*(i+1)]
            img_patches =      debug_dic['patches'][n_rows*i:n_rows*(i+1)]
            kept_idxs =      debug_dic['kept_idxs'][n_rows*i:n_rows*(i+1)]
            masked_patches = get_masked_img_in_patch_form(img_patches, kept_idxs)
            def get(img):
                img = patches_to_imgs_chw_slow(img, self.n_patches_in_row, self.patch_size)
                img = reverse_normalize_transform(torch.cat(tuple(img), dim=1))
                return img
            imgs.append( torch.cat((
                get(img_patches),
                get(pred_patches),
                get(masked_patches),
            ), dim=2))
        img = torch.cat(imgs, dim=2)
        writer.add_image('imgs', img.detach().cpu(), step)




def drop_tokens(patches, ratio):
    '''
    Patch ordering will not be preserved!
    Assumes positional encoding is already applied.
    '''
    b, seq, c = patches.shape
    n_keep = math.ceil( (1- ratio) * patches.shape[1])
    idxs2d = torch.cat([torch.randperm(seq)[None] for _ in range(b)]).to(patches.device)
    idxs = idxs2d[:, :, None].expand(-1, -1, c)
    shuffled_patches = torch.gather(patches, dim=1, index=idxs) 
    return shuffled_patches[:, :n_keep], idxs2d[:, :n_keep], idxs2d[:, n_keep:]

def get_masked_img_in_patch_form(patches, kept_idxs):
    b, n, c = patches.shape
    zeroed = torch.zeros((b*n, c), dtype=patches.dtype, device=patches.device)
    patches = patches.reshape(b*n, c) # whyyyy
    offset = (torch.arange(0, b) * n)[:, None].to(patches.device)
    kept_idxs = (kept_idxs + offset).view(-1)
    zeroed[kept_idxs] = patches[kept_idxs]
    return zeroed.view(b, n, c)

def _get_conv_weight(h, w):
    zeros = []
    for i in range(3*h*w):
        z = th.zeros((3*h*w,))
        z[i] = 1
        zeros.append(z.reshape(1, 3, h, w))
    return th.cat(zeros)

def imgs_to_patches(imgs, patch_size):
    '''
    returns b, -1, 3 x patch_size x patch_size
    '''
    weight = _get_conv_weight(patch_size, patch_size).to(imgs.device)
    convolved = F.conv2d(imgs, weight, None, patch_size)
    return convolved.view(imgs.shape[0], 3*patch_size*patch_size, -1).permute(0, 2, 1)

def visualize_patches(patches, n_patches_in_row, patch_size):
    assert len(patches.shape) == 2, 'give only one patch'
    patches = patches.view(-1, 3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1)
    patches_in_column = patches.shape[0] // n_patches_in_row

    for i, patch in enumerate(patches):
        plt.subplot(n_patches_in_row, patches_in_column, i+1)
        plt.imshow(patch.numpy())

    return patches

def patches_to_imgs_chw_slow(patches, n_patches_in_row, patch_size):
    # patches.shape = b, n, 3*patch_size**2
    b, n, _ = patches.shape
    patches = patches.view(b, n//n_patches_in_row, n_patches_in_row, 3, patch_size, patch_size)
    imgs = []
    for i in range(patches.shape[1]):
        row = []
        for j in range(patches.shape[2]):
            row.append( patches[:, i, j])
        row = torch.cat(row, dim=-1)
        imgs.append(row)
    imgs = torch.cat(imgs, dim=-2)
    return imgs

def patches_to_imgs_hwc_slow(patches, n_patches_in_row, patch_size):
    return patches_to_imgs_chw_slow(patches, n_patches_in_row, patch_size).permute(0, 2, 3, 1)




# %% 
if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    import matplotlib.pyplot as plt
    dataset = CIFAR10('../data', True, transform=T.ToTensor())
    patch_size = 4
    imgs = dataset[100][0][None]
    _, _, h, w = imgs.shape
    n_patches_in_row = w // patch_size
    patches = imgs_to_patches(imgs, patch_size)
    patches = drop_tokens(patches, 7)
    print(patches.shape)

    
    # visualize_patches(patches[0], n_patches_in_row, patch_size)
    imgs2 = patches_to_imgs_hwc_slow(patches, n_patches_in_row, patch_size)

    # run cell with MAEModel in it for this line to work
    model = MAEModel()
    processor = MaskedAutoencoderProcessor()
    _, dic = processor.process_batch(model, {'a':imgs}, device='cpu')
    
    img = patches_to_imgs_hwc_slow(dic['predictions'], n_patches_in_row, patch_size)[0]
    plt.imshow(img.detach().cpu().numpy())




# %%
import torch
a = torch.arange(0, 10).view(2, 5)

a
# %%
