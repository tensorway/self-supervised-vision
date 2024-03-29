from torchvision import transforms
from torchvision import transforms as T 

    
easy_transform = transforms.Compose([
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

reverse_normalize_transform = transforms.Compose([
    transforms.Normalize((0.0, 0.0, 0.0), (2, 2, 2)),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])