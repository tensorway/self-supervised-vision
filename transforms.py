from torchvision import transforms
from torchvision import transforms as T 

hard_transform = transforms.Compose([
    transforms.Resize(48),
    transforms.RandomCrop(32),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.RandomGrayscale(p=0.3),
    transforms.RandomSolarize(threshold=200, p=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    
train_transform = transforms.Compose([
    transforms.Resize(32),
    # transforms.RandomCrop(32),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    # transforms.RandomGrayscale(p=0.3),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomSolarize(threshold=200, p=0.3),
    # transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])



# class InterpolationMode(Enum):
#     """Interpolation modes
#     Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
#     """

#     NEAREST = "nearest"
#     BILINEAR = "bilinear"
#     BICUBIC = "bicubic"
#     # For PIL compatibility
#     BOX = "box"
#     HAMMING = "hamming"
#     LANCZOS = "lanczos"


# class AugMix:
#     '''
#     implementation of augmix augmentation 
#     paper = https://openreview.net/forum?id=S1gmrxHFvB

#     needs a list of transforms from which to randomly
#     choose and mix, the default is from augmix paper.
#     contrast, color, brightness, sharpness, and Cutout were not 
#     in the paper since it was studying the robustness and those
#     operations were in the test set. Those transformations are 
#     included here
#     '''
#     def __init__(self, 
#         transforms_list = [
#             T.RandomAffine(degrees=0, interpolation=InterpolationMode.BILINEAR),
#             T.RandomAffine(interpolation=InterpolationMode.BILINEAR),
#             T.RandomRotation( interpolation=InterpolationMode.BILINEAR )
#             ], 
#         mixing_coefficient=1, 
#         num_paths=3
#         ) -> None:
#         self.transforms_list = transforms_list
#         self.mixing_coefficient = mixing_coefficient
#         self.num_paths = num_paths

#     def __call__(self, tensor_img):
#         xaug = th.zeros_like(tensor_img)
#         concentration = [self.mixing_coefficient for _ in range(len(self.trainsforms_list))]
#         mixing_weights = th.distributions.Dirichlet(concentration).sample(self.num_paths)
#         for path_weight in mixing_weights:
#             num_ops = random.randint(1, 3)
#             ops = np.random.choice(self.transforms_list, size=num_ops, replace=False)
#             augmented_img = tensor_img
#             for op in ops:
#                 augmented_img = op(augmented_img)
#             xaug = path_weight*augmented_img
#         skip_weight = th.distributions.Beta(self.mixing_coefficient, self.mixing_coefficient).sample(1)
#         return tensor_img*skip_weight + xaug*(1-skip_weight)

#     def __repr__(self) -> str:
#         format_string = self.__class__.__name__ + "("
#         for t in self.transforms_list:
#             format_string += "\n"
#             format_string += f"    {t}"
#         format_string += "\n)"
#         return format_string
