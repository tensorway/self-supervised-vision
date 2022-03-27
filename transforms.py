from torchvision import transforms


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